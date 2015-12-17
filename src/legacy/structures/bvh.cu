#include <bvh.h>

unsigned long long sibling_cnt =1;
unsigned long long global_mortonCode=1 ;

//#####################################################################
//######################## IMPLEMENTATION #############################
//#####################################################################

void InsertDataBVH(bitmask_t* keys, BVH_Branch* data)
{

  char dataFileName[100];

  if( !strcmp(DATATYPE, "high" ) )
    sprintf(dataFileName, "/home/jwkim/inputFile/%s_dim_data.%d.bin\0", DATATYPE, NUMDIMS); 
  else
    sprintf(dataFileName, "/home/jwkim/inputFile/NOAA%d.bin",myrank); 

  FILE *fp = fopen(dataFileName, "r");
  if(fp==0x0){
    printf("Line %d : Insert file open error\n", __LINE__);
    printf("%s\n",dataFileName);
    exit(1);
  }


  BVH_Branch b;
  float f_read;
  for(int i=0; i<NUMDATA; i++){

    for(int j=0;j<NUMDIMS;j++){
      fread( &f_read, sizeof(float), 1, fp);
      b.setRectBoundary(j,f_read),
      b.setRectBoundary(NUMDIMS+j, b.getRectBoundary(j));
    }

//    printf("%.6f, %.6f, %.6f\n", b.rect.boundary[0], b.rect.boundary[1], b.rect.boundary[2]); 
//    printf("%d, %d, %d\n", b.rect.boundary[0], b.rect.boundary[1], b.rect.boundary[2]); 
    //unsigned int t_code = morton3D(b.getRectBoundary(0), b.getRectBoundary(1), b.getRectBoundary(2));
    /*
    
    unsigned long long  input[NUMDIMS];
    for(int j=0;j<NUMDIMS;j++){
      input[j] = b.rect.boundary[j]*1000000; 
    }

    //If some nodes have same morton code, we may lose some hit count
    //because we may think that all of them have been checked,
    //but often, only some of them has been checked and we can set a passed_morton_code as same morton code
    //and then just skip the others
    
    //unsigned long long t_code = mortonEncode_magicbits(input[0], input[1], input[2]);
    unsigned long long  t_code = mortonEncode_magicbits2(input, NUMDIMS);
    */
    unsigned long long  t_code = 0;

    b.setMortonCode( t_code );

    keys[i] = b.getMortonCode();;
    data[i] = b;
  }

  if(fclose(fp) != 0){
    printf("Line %d : Insert file close error\n", __LINE__);
    exit(1);
  }
}
int MortonCompare(const void* a, const void* b)
{
  BVH_Branch* b1 = (BVH_Branch*) a;
  BVH_Branch* b2 = (BVH_Branch*) b;

  return (b1->getMortonCode() > b2->getMortonCode()) ? 1 : 0;
}

void Build_BVH(int index_type)
{

//  printf("####################################################\n");
//  printf("################# INSERT DATA ######################\n");
//  printf("####################################################\n");

  //Measure the time with these variables
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  float elapsed_time;

  //Insert data and calculate the Hilbert value 
  bitmask_t *keys = (bitmask_t*)malloc(sizeof(bitmask_t)*NUMDATA);
  BVH_Branch *data = (BVH_Branch*) malloc( sizeof(BVH_Branch)*NUMDATA);
  cudaEventRecord(start_event, 0);

  InsertDataBVH(keys, data);

  cudaEventRecord(stop_event, 0) ;
  cudaEventSynchronize(stop_event) ;
  cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

  printf("Insertion time = %.3fs\n\n", elapsed_time/1000.0f);

//  print_BVHBranch(data);
//  print_BVHBranch(&data[NUMDATA-1]);
  

//  printf("####################################################\n");
//  printf("############# SORT THE DATA ON GPU #################\n");
//  printf("####################################################\n");

  cudaEventRecord(start_event, 0);

  //copy host to device
  thrust::device_vector<bitmask_t> d_keys(NUMDATA);
  thrust::device_vector<BVH_Branch> d_data(NUMDATA);
  thrust::copy(keys, keys+NUMDATA, d_keys.begin() );
  thrust::copy(data, data+NUMDATA, d_data.begin() );

  //sort on GPU
  thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_data.begin() );

  thrust::copy(d_data.begin(), d_data.end(), data );

  cudaEventRecord(stop_event, 0) ;
  cudaEventSynchronize(stop_event) ;
  cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
  printf("Sort Time on GPU = %.3fs\n\n", elapsed_time/1000.0f);

  //free()
  thrust::device_vector<bitmask_t>().swap(d_keys);
  thrust::device_vector<BVH_Branch>().swap(d_data);

  //DEBUG
  /*
  printf("sorted morton code ---\n");  
  for(int i=0; i<NUMDATA-1; i++)
  {
    printf("%lu ", data[i].mortonCode);
    if( i % 10 == 0 )
      printf("\n");
    //if( data[i].mortonCode > data[i+1].mortonCode)
    //  printf("%lu ", data[i].mortonCode);
  }
  printf("sorted morton code ---\n");  
  */

  if( POLICY ){
    BVH_Branch temp[NUMDATA];
    memcpy( temp, data, sizeof(BVH_Branch)*NUMDATA);

    int seq[PARTITIONED];

    int boundary  = NUMDATA%PARTITIONED;

    // there are trees which have different tree heights
    int partitioned_NUMDATA[2];
    partitioned_NUMDATA[0] = partitioned_NUMDATA[1] = (NUMDATA/PARTITIONED);
    if( boundary )
      partitioned_NUMDATA[0]++;

    int tree_idx = 0;
    int offset = 0 ;
    for(int i=0; i<PARTITIONED; i++)
    {
      if( i == boundary )
        tree_idx = 1;

      seq[i] = offset;
      offset += partitioned_NUMDATA[tree_idx];
      //printf("seq offset = %d\n",seq[i]);
    }

    for(int i=0; i<NUMDATA; i++)
    {
      int partition_idx =  i%PARTITIONED;
      data[seq[partition_idx]] = temp[i];
      seq[partition_idx]++;
    }
  }

  //####################################################
  //############ BUILD UP   BVH Trees ##################
  //####################################################

  int bound = ceil ( ( double) NUMDATA/PARTITIONED );
  for( int p=0; p<PARTITIONED; p++)
  {
    unsigned long long totalNodes=0;
    int tree_height=0;
    //printf("Build up BVH trees\n" );

    sibling_cnt = 1;
    global_mortonCode = 1;

    if( DEBUG )
      printf("bound start from  %d to %d\n", bound*p,  min( ( bound*(p+1))-1,NUMDATA-1));

    ch_root[p] = (char *) generateHierarchy(data, bound*p /*first*/ , min( ( bound*(p+1))-1,NUMDATA-1) /*last*/, 0 /*level*/, &totalNodes, &tree_height );

    int numOfnodes[tree_height];
    memset( numOfnodes, 0, sizeof(int)*tree_height);

    if( DEBUG )
      printf("settingBVH_trees\n" );

    settingBVH_trees((BVH_Node*)ch_root[p], numOfnodes, tree_height-1); //setting level and number of nodes per each level


    if( p == 0 || p+1 == PARTITIONED )
    {
      printf("%d tree height  %d\n",p, tree_height);
      for(int i=tree_height-1; i>=0; i--)
        printf("%d number of nodes %d\n", i, numOfnodes[i]);
      printf("total nodes %d\n", totalNodes);
    }


    
    numOfleafNodes = numOfnodes[0];

    if( DEBUG )
    {
      printf("traverseBVH_BFS\n" );
      traverseBVH_BFS((BVH_Node*) ch_root[0] );
    }

    unsigned long long indexSize = sizeof(BVH_Node)*totalNodes;
    char* buf = (char*)malloc(indexSize);

    if( DEBUG )
      printf("BVHTreeDumpToMem\n" );

    BVHTreeDumpToMem( (BVH_Node*)ch_root[p],  buf, tree_height); 

    if( index_type == 2 && METHOD[6] == true)
    {
      unsigned long long off = 0;
      char* buf2 = (char*)malloc(indexSize);


      if( DEBUG )
        printf("BVHTreeDumpToMemDFS\n" );
      BVHTreeDumpToMemDFS( buf, buf2, 0, &off );

      if( DEBUG )
        printf("LinkUpSibling2\n" );
      LinkUpSibling2( buf2, totalNodes); // skip pointer

      //  printf("checkDumpedTree\n" );
      //  checkDumpedTree(buf2, totalNodes); 

      memcpy(buf, buf2, indexSize);
      free(buf2);
    }
    else
    {
      if( DEBUG )
        printf("LinkUpSibling\n" );
      LinkUpSibling( buf, tree_height, totalNodes);
      //  printf("checkDumpedTree\n" );
      //  checkDumpedTree(buf, totalNodes); 
    }

    /*
    float area = .0f;
    BVH_Node* node = (BVH_Node*)buf;
//    checkNodeOverlap( node, &area);
//    printf("Intersected area is %5.5f\n", area);
    
    int overlapCnt = 0;
    int nodeCnt = 0;
    for( int i=0; i< totalNodes; i++)
    {
      if( node->level > 0 )
      {
        nodeCnt += node->count;

        for(int i1 = 0; i1<node->count-1; i1++)
        {
          for(int i2 = i1+1; i2<node->count; i2++)
          {
            if( RectOverlap(&node->branch[i1].rect, &node->branch[i2].rect) )
            {
              area += IntersectedRectArea(&node->branch[i1].rect, &node->branch[i2].rect);
              overlapCnt++;
            }
          }
        }
      }
      node++;
    }
    printf("node count %d overlap cnt %d Intersected area is %5.5f\n", nodeCnt, overlapCnt, area);
    */


    //printf("globalBVHTreeLoadFromMem\n" );

    char* d_treeRoot;
    cudaMalloc( (void**) & d_treeRoot, indexSize+BVH_PGSIZE); // to store stopNode for SkipPointer 
    cudaMemcpy(d_treeRoot, buf, indexSize, cudaMemcpyHostToDevice);
    globalBVHTreeLoadFromMem<<<1,1>>>(d_treeRoot, p, NUMBLOCK, PARTITIONED, totalNodes );

    if( DEBUG )
    {
      printf("Print BVH\n");
      if( p == 0 )
        global_print_BVH<<<1,1>>>(p, 1);
      cudaThreadSynchronize();
    }

    if( index_type == 2 && METHOD[6] == true) 
    {
      globalBVH_Skippointer<<<1,1>>>(p, tree_height, totalNodes);
      globaltranspose_BVHnode<<<1,NODECARD>>>(p, totalNodes+1);
    }
    else
      globaltranspose_BVHnode<<<1,NODECARD>>>(p, totalNodes);
    
    cudaThreadSynchronize();

    free(buf);
  }

  size_t avail, total;
  cudaMemGetInfo( &avail, &total );
  size_t used = total-avail;
  //DEBUG
  printf(" Used %lu / Total %lu ( %.2f % ) \n\n\n",used,total, ( (double)used/(double)total)*100);



  //Need to debug
//  BVH_RecursiveSearch();
   free(keys);
   free(data);
}


// BVH
int findSplit( BVH_Branch *data, int first, int last)
{
  // Identical Morton codes => split the range in the middle.
  //
  unsigned long long firstCode = data[first].getMortonCode();
  unsigned long long lastCode = data[last].getMortonCode();

  if (firstCode == lastCode)
    return (first + last) >> 1;


  // Calculate the number of highest bits that are the same
  // for all objects, using the count-leading-zeros intrinsic.

  int commonPrefix = __builtin_clzl(firstCode ^ lastCode);

  // Use binary search to find where the next bit differs.
  // Specifically, we are looking for the highest object that
  // shares more than commonPrefix bits with the first one.
  
  int split = first; // initial guess
  int step = last - first;

  do
  {
    step = (step + 1) >> 1; // exponential decrease
    int newSplit = split + step; // proposed new position

    if (newSplit < last)
    {
      unsigned long long splitCode = data[newSplit].getMortonCode();
      int splitPrefix = __builtin_clzl(firstCode ^ splitCode);

      if (splitPrefix > commonPrefix)
        split = newSplit; // accept proposal
    }
  }
  while (step > 1);

  return split;
}


void findSplits(BVH_Branch* data, int first, int last, int* split_pos, int *split_pos_cnt )
{
  if ((last-first) < NODECARD)
    return ;
  if(*split_pos_cnt >= (NODECARD-1)) return;

  //printf("first %d last %d \n", first, last);
  int split;
if( BUILD_TYPE == 1)
  split = findSplit(data, first, last);
else 
  split = (first+last)/2;

  //printf("split %d \n", split );
  split_pos[(*split_pos_cnt)++] = split; 


  if(*split_pos_cnt >= (NODECARD-1)) return;

  // enqueue the split position
  if( ( split - first +1) > NODECARD)
  {
    //printf("first %d split %d \n", first, split);
    struct split_range *left = (struct split_range*)malloc(sizeof(struct split_range));
    left->leftbound = first;
    left->rightbound = split;
    sq.push(left);
  }

  if( ( last - split+2 ) > NODECARD)
  {
    //printf("split+1 %d last %d \n", split+1, last);
    struct split_range *right = (struct split_range*)malloc(sizeof(struct split_range));
    right->leftbound = split+1;
    right->rightbound = last;
    sq.push(right);
  }

  
  //dequeue the 
  if( !sq.empty())
  {
    //printf("DO\n");
    struct split_range *temp = sq.front();
    sq.pop();

    findSplits(data, temp->leftbound, temp->rightbound, split_pos, split_pos_cnt );
  }
}

BVH_Node* generateHierarchy( BVH_Branch* data, int first, int last, int level, unsigned long long *totalNodes, int *tree_height )
{
  int split_pos[NODECARD];
  int split_pos_cnt=0;

  // create a leaf node.
  if ( (last-first) < NODECARD )
  {
    BVH_Node* node = new BVH_Node((last-first)+1, INT_MAX);
    for(int i=0; i<node->getCount(); i++){
      node->setBranch(i, data[first+i]);
      node->setBranchChild(i,NULL);
      node->branch[i].mortonCode = global_mortonCode++;
    }
    (*totalNodes)++;
    *tree_height = max (level+1 , *tree_height);

    //node's sibling
    char* tmp = (char*) node;
    memcpy(tmp+8+(16+8*NUMDIMS)*NODECARD, &sibling_cnt, sizeof(unsigned long long)); 
    sibling_cnt++;
  
    return node;
  }


  // Determine where to split the range.
  if( BUILD_TYPE == 3)  { //  real data(3-dim)
    int t_num = last-first+1;
    //sort the data by current level;
    if( level % 3 == 0 )  {//sort by x axis
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d0);
    } else if( level %3 == 1)  {// sort by y axis
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d1);
    } else  {// sort by z axis
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d2);
    }

    //find a split position 
    /*
    int boundary= t_num%NODECARD;
    int num[2];
    num[0] = num[1] = (t_num/NODECARD);
    if( boundary )
      num[0]++;

    int offset = first;
    int inc = num[0];
    for( int n=0; n<NODECARD-1; n++)
    {
      if( n == boundary )
        inc = num[1];
      offset += inc;
      split_pos[n] = offset-1;
      split_pos_cnt++;
    }
    */
  }else if( BUILD_TYPE == 4)
  {
    int t_num = last-first+1;
  
    //choose random dimension for sorting
    int chosen_dimensioin = rand()%NUMDIMS;
//    printf("chosen dimension for sorting is %d\n", chosen_dimensioin);

    //sort the data by current level;
    if( chosen_dimensioin == 0 )
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d0);
    else if( chosen_dimensioin == 1) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d1);
    else if( chosen_dimensioin == 2) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d2);
    else if( chosen_dimensioin == 3) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d3);
#if NUMDIMS > 4
    else if( chosen_dimensioin == 4) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d4);
    else if( chosen_dimensioin == 5) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d5);
    else if( chosen_dimensioin == 6) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d6);
    else if( chosen_dimensioin == 7) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d7);
    else if( chosen_dimensioin == 8) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d8);
    else if( chosen_dimensioin == 9) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d9);
    else if( chosen_dimensioin == 10) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d10);
    else if( chosen_dimensioin == 11) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d11);
    else if( chosen_dimensioin == 12) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d12);
    else if( chosen_dimensioin == 13) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d13);
    else if( chosen_dimensioin == 14) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d14);
    else if( chosen_dimensioin == 15) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d15);
#if NUMDIMS > 16 
    else if( chosen_dimensioin == 16) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d16);
    else if( chosen_dimensioin == 17) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d17);
    else if( chosen_dimensioin == 18) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d18);
    else if( chosen_dimensioin == 19) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d19);
    else if( chosen_dimensioin == 20) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d20);
    else if( chosen_dimensioin == 21) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d21);
    else if( chosen_dimensioin == 22) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d22);
    else if( chosen_dimensioin == 23) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d23);
    else if( chosen_dimensioin == 24) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d24);
    else if( chosen_dimensioin == 25) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d25);
    else if( chosen_dimensioin == 26) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d26);
    else if( chosen_dimensioin == 27) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d27);
    else if( chosen_dimensioin == 28) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d28);
    else if( chosen_dimensioin == 29) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d29);
    else if( chosen_dimensioin == 30) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d30);
    else if( chosen_dimensioin == 31) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d31);
    else if( chosen_dimensioin == 32) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d32);
    else if( chosen_dimensioin == 33) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d33);
    else if( chosen_dimensioin == 34) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d34);
    else if( chosen_dimensioin == 35) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d35);
    else if( chosen_dimensioin == 36) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d36);
    else if( chosen_dimensioin == 37) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d37);
    else if( chosen_dimensioin == 38) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d38);
    else if( chosen_dimensioin == 39) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d39);
    else if( chosen_dimensioin == 40) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d40);
    else if( chosen_dimensioin == 41) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d41);
    else if( chosen_dimensioin == 42) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d42);
    else if( chosen_dimensioin == 43) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d43);
    else if( chosen_dimensioin == 44) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d44);
    else if( chosen_dimensioin == 45) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d45);
    else if( chosen_dimensioin == 46) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d46);
    else if( chosen_dimensioin == 47) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d47);
    else if( chosen_dimensioin == 48) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d48);
    else if( chosen_dimensioin == 49) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d49);
    else if( chosen_dimensioin == 50) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d50);
    else if( chosen_dimensioin == 51) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d51);
    else if( chosen_dimensioin == 52) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d52);
    else if( chosen_dimensioin == 53) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d53);
    else if( chosen_dimensioin == 54) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d54);
    else if( chosen_dimensioin == 55) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d55);
    else if( chosen_dimensioin == 56) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d56);
    else if( chosen_dimensioin == 57) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d57);
    else if( chosen_dimensioin == 58) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d58);
    else if( chosen_dimensioin == 59) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d59);
    else if( chosen_dimensioin == 60) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d60);
    else if( chosen_dimensioin == 61) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d61);
    else if( chosen_dimensioin == 62) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d62);
    else if( chosen_dimensioin == 63) 
      qsort(data+first, t_num, sizeof(BVH_Branch), comp_d63);
    else
      printf("????\n");
#endif
#endif
  }

  findSplits(data, first, last, split_pos, &split_pos_cnt ); // get an array of split_pos

  /* Free the entire tail queue. */
  while (!sq.empty()) {
    sq.pop();
  }

  //setting default split position
  split_pos[split_pos_cnt++]=last;

  qsort(split_pos, split_pos_cnt, sizeof(int), comp); 

  /*
  if( DEBUG )
  {
    printf("Split pos ");
    for( int i =0; i < 1; i++)
      printf("%d ", split_pos[i]);
    printf("\n\n");
  }
  */
  

  // Process the resulting sub-ranges recursively.

  BVH_Node* childNodes[split_pos_cnt];

  int leftbound = first;
  for(int i=0; i<split_pos_cnt; i++){
      childNodes[i] = generateHierarchy(data, leftbound, split_pos[i], level+1 , totalNodes,  tree_height);
      leftbound = split_pos[i]+1;

  }

  BVH_Node* node = new BVH_Node(split_pos_cnt, level);


  for(int i=0; i<node->getCount(); i++)
  {
    node->setBranchMortonCode(i,childNodes[i]->getMaxMortonCode());
    node->setBranchChild(i,childNodes[i]);

    //Find out the min, max boundaries in this node and set up the parent rect.
    float MIN_boundary = FLT_MAX;
    float MAX_boundary = FLT_MIN;
    for( int d = 0 ; d < NUMDIMS ; d ++ )
    {
      MIN_boundary = FLT_MAX;
      MAX_boundary = FLT_MIN;

      for( int c = 0 ; c < childNodes[i]->getCount() ; c ++ )
      {
        if( MIN_boundary > childNodes[i]->getBranchRectBoundary(c,d))
          MIN_boundary = childNodes[i]->getBranchRectBoundary(c,d);
        if( MAX_boundary < childNodes[i]->getBranchRectBoundary(c,d+NUMDIMS))
          MAX_boundary = childNodes[i]->getBranchRectBoundary(c,d+NUMDIMS);
      }
      node->branch[i].rect.boundary[d] = MIN_boundary;
      node->branch[i].rect.boundary[d+NUMDIMS] = MAX_boundary;
      //node->setBranchRectBoundary(i, d, MIN_boundary);
      //node->setBranchRectBoundary(i, d+NUMDIMS, MAX_boundary);
    }
  }
  (*totalNodes)++;

  char* tmp = (char*) node;
  memcpy(tmp+8+(16+8*NUMDIMS)*NODECARD, &sibling_cnt, sizeof(unsigned long long)); 
  sibling_cnt++;
  

  return node;
}

void BVHTreeDumpToMem(BVH_Node* n, char *buf, int tree_height) // Breadth-First Search
{
  while(!bvh_q.empty()){
    bvh_q.pop();
  }

  unsigned long long nodeSeq = 0;
  unsigned long long off = 0;

  bvh_q.push(n);
  nodeSeq++;

  while(!bvh_q.empty()){
    BVH_Node* t = bvh_q.front();
    bvh_q.pop();

    memcpy(buf+off, t, BVH_PGSIZE);

    if (t->getLevel() > 0){ // this is an internal node in the tree
      for (int i=0; i<t->getCount(); i++){
        BVH_Branch b = t->branch[i];
        bvh_q.push(b.child);
        unsigned long long branch_off =  nodeSeq * BVH_PGSIZE ;

        nodeSeq++;
        memcpy(buf+off+8+((8+8*NUMDIMS)*(i+1))+(8*i), &branch_off, sizeof(unsigned long long));
      }
    }
    off += BVH_PGSIZE;
  }
}
unsigned long long BVHTreeDumpToMemDFS(char* buf, char* buf2, unsigned long long childOff, unsigned long long *off )
{

             
  //debug
  /*
  unsigned long long oldOff = *off;
  if( *off >= 966966936){
    printf("%lu\n", *off);
    BVH_Node* node = (BVH_Node*) (buf+childOff);
    printf("%d\n",__LINE__);

    printf("%d\n",__LINE__);

    printf("%lu\n",buf2);
    printf("%lu\n",oldOff);
    printf("%lu\n",buf2+oldOff);
    printf("%lu\n",node);
    BVH_Node* node2 = (BVH_Node*) (buf2+oldOff);
    printf("%lu\n",node2);
    printf("n l %d\n", node->level);
    printf("n2 l %d\n", node2->level);

    memcpy(buf2+oldOff, node, BVH_PGSIZE);
    printf("%d\n",__LINE__);
    *off += BVH_PGSIZE;

    if( node->level > 0)
    {
      for( int i=0; i<node->count; i++)
      {
        printf("%d\n",__LINE__);
        unsigned long long branch_off = BVHTreeDumpToMemDFS( buf, buf2, (unsigned long long)node->branch[i].child , off);
        printf("%d\n",__LINE__);
        memcpy(buf2+oldOff+8+((8+8*NUMDIMS)*(i+1))+(8*i), &branch_off, sizeof(unsigned long long));
        printf("%d\n",__LINE__);
      }
    }
  }
  else

  {
  */
    BVH_Node* node = (BVH_Node*) (buf+childOff);

    unsigned long long oldOff = *off;
    memcpy(buf2+oldOff, node, BVH_PGSIZE);
    *off += BVH_PGSIZE;

    if( node->level > 0)
    {
      for( int i=0; i<node->count; i++)
      {
        unsigned long long branch_off = BVHTreeDumpToMemDFS( buf, buf2, (unsigned long long)node->branch[i].child , off);
        memcpy(buf2+oldOff+8+((8+8*NUMDIMS)*(i+1))+(8*i), &branch_off, sizeof(unsigned long long));
      }
    }
  //}

  return oldOff;
}


void LinkUpSibling( char* buf, int tree_height, int totalNodes)
{

  unsigned long long sibling_cnt_perLevel[tree_height];
  memset(sibling_cnt_perLevel, 0, sizeof(int)*tree_height);

  unsigned long long sibling_array[tree_height][totalNodes];
  memset(sibling_array, 0, sizeof(unsigned long long)*tree_height*totalNodes);

  BVH_Node* n = (BVH_Node*) buf;
  unsigned long long off = 0;

  for( int i=0; i<totalNodes; i++, n++)
  {
//    printf("%d %lu\n", n->level,(long) n->sibling);
    sibling_array[n->level][(unsigned long long)(n->sibling)] = off;
    sibling_cnt_perLevel[n->level]++;
    off += BVH_PGSIZE;
  }

  unsigned long long  i, j, cnt;
  for(int h=0; h<tree_height; h++)
  {
    i=cnt=0;
    while(cnt<sibling_cnt_perLevel[h])
    {
      while( sibling_array[h][i] == 0 && i< (totalNodes-1)  )
      {
        i++;
//        printf("i  %d\n",i);
      }
      j=i+1;
      while( sibling_array[h][j] == 0 && j < totalNodes )
      {
        j++;
//        printf("j  %d\n",j);
      }
      if( i+1 < totalNodes && j < totalNodes ) 
      {
//        printf("i  %d, j  %d\n",i,j);
        //link up sibling if we found
        unsigned long long s_value  = sibling_array[h][j];
        unsigned long long tmp = sibling_array[h][i];
//        printf("s_value %lu\n",s_value);
//        printf("tmp %lu\n",sibling_array[h][i]);
        memcpy(buf+tmp+8+(16+8*NUMDIMS)*NODECARD, &s_value, sizeof(unsigned long long)); 
        i=j;
        cnt++;

      }
      else
      {
        unsigned long long s_value  = 0;
        unsigned long long tmp = sibling_array[h][i];
//        printf("single value %lu\n",s_value);
//        printf("single node%lu\n",sibling_array[h][i]);
        memcpy(buf+tmp+8+(16+8*NUMDIMS)*NODECARD, &s_value, sizeof(unsigned long long)); 
        break;
      }
    }
    
  }
}
void LinkUpSibling2( char* buf, int totalNodes)
{
  BVH_Node* n = (BVH_Node*) buf;

  unsigned long long n_pointer = 0;
  memcpy(buf+8+(16+8*NUMDIMS)*NODECARD, &n_pointer, sizeof(unsigned long long)); 

  for( int i=0; i<totalNodes-1; i++, n++)
  {
    if( n->level > 0 )
    {
      for(int j=0; j<n->count-1; j++)
      {
        unsigned long long child = (unsigned long long)n->branch[j].child;
        unsigned long long next_child = (unsigned long long)n->branch[j+1].child;
        memcpy(buf+child+8+(16+8*NUMDIMS)*NODECARD, &next_child, sizeof(unsigned long long)); 
        memcpy(buf+next_child+8+(16+8*NUMDIMS)*NODECARD, &n_pointer, sizeof(unsigned long long)); 
      }
    }
  }

}
//setting level and number of nodes per each level
void  settingBVH_trees(BVH_Node* n, int *numOfnodes, const int reverse )
{
  while(!bvh_q.empty()){
    bvh_q.pop();
  }

  bvh_q.push(n);

  while(!bvh_q.empty()){
    BVH_Node* t = bvh_q.front();
    bvh_q.pop();

    if( t->getLevel() == INT_MAX)
    {
      t->setLevel(0);
    }
    else 
    {
      t->setLevel(reverse-(t->getLevel()));
    }

    numOfnodes[t->getLevel()]++;

    if( t->getLevel() > 0 )
    {
      for (int i=0; i<NODECARD; i++){
        BVH_Branch b = t->getBranch(i);
        if( b.getChild() != NULL )
          bvh_q.push(b.getChild());
      }
    }
  }
}
void  settingBVH_trees2(BVH_Node* n, int *numOfnodes, const int reverse, int totalNodes )
{
  for(int i=0; i<totalNodes; i++)
  {

    if( n->getLevel() == INT_MAX )
    {
      n->setLevel(0);
    }
    else
    {
      n->setLevel(reverse-(n->getLevel()));
    }
    numOfnodes[n->getLevel()]++;

  }
}
void traverseBVH_BFS(BVH_Node* n) // Breadth-First Search
{
  while(!bvh_q.empty()){
    bvh_q.pop();
  }

  bvh_q.push(n);

  while(!bvh_q.empty()){
    BVH_Node* t = bvh_q.front();
    bvh_q.pop();

    printf("morton code %lu\n",t->getMaxMortonCode());
    printf("node %lu\n",t);
    printf("# of childs %d\n",t->getCount());
    if( t->getLevel() > 0 )
    {
      for( int i=0; i<t->getCount(); i++)
        printf("%d child %lu\n", i, t->getBranch(i).getChild());
    }
    printf("level %d\n",t->getLevel());
    printf("\n\n");

    if( t->getLevel() > 0 )
    {
      for (int i=0; i<t->getCount(); i++){
        BVH_Branch b = t->getBranch(i);
        bvh_q.push(b.getChild());
      }
    }
  }
}

void checkDumpedTree(char* buf, int totalNodes)
{
  printf("====START check Dumpted Tree====\n");
  BVH_Node* n = (BVH_Node*) buf;
  unsigned long long off = 0;

  for( int i=0; i<totalNodes; i++, n++)
  {
    printf("morton code hIndex %lu\n",n->getMaxMortonCode());
    printf("n %lu\n",off);
    printf("# of childs %d\n",n->getCount());
    if( n->getLevel() > 0)
    {
      for( int j=0; j<n->getCount(); j++)
        printf("%d child %lu\n", j, n->getBranch(j).getChild());
    }
    printf("sibling %lu\n", n->sibling);
    printf("level %d\n",n->getLevel());
    printf("\n\n");
    off += BVH_PGSIZE;
  }


  printf("====END OF check Dumpted Tree====\n");
}

void print_BVHNode(BVH_Node* n )
{
    printf("morton code hIndex %lu\n",n->getMaxMortonCode());
    printf("n %lu\n",n);
    printf("# of childs %d\n",n->getCount());
    if( n->getLevel() > 0)
    {
      for( int j=0; j<n->getCount(); j++)
        printf("%d child %lu\n", j, n->getBranch(j).getChild());
    }
    printf("sibling %lu\n", n->sibling);
    printf("level %d\n",n->getLevel());
    printf("\n\n");
}

void print_BVHBranch(BVH_Branch* n )
{
    printf("n %lu\n",n);
    for( int d = 0; d<NUMDIMS; d++)
      printf(" %.5f \n",n->rect.boundary[d]);
    printf("child %lu\n", n->getChild());
    printf("\n\n");
}



__global__ void globalBVHTreeLoadFromMem(char *buf, int partition_no, int NUMBLOCK, int PARTITIONED /* number of partitioned index*/, int numOfnodes)
{
  if( partition_no == 0 ){
    devNUMBLOCK = NUMBLOCK;
    deviceBVHRoot = (BVH_Node**) malloc( sizeof(BVH_Node*) * PARTITIONED );
  }

  deviceBVHRoot[partition_no] = (BVH_Node*) buf;
  devBVHTreeLoadFromMem(deviceBVHRoot[partition_no], buf, numOfnodes);
}

__device__ void devBVHTreeLoadFromMem(BVH_Node* n, char *buf, int numOfnodes)
{

  n->parent = NULL;
  for( int i = 0; i < numOfnodes; i ++){
    if( n->level > 0 ){
      for( int c = 0 ; c < n->count; c++){
        n->branch[c].child = (BVH_Node*) ((size_t) n->branch[c].child + (size_t) buf);
        n->branch[c].child->parent = n;
      }
    }
    if( n->sibling == 0)
      n->sibling = NULL;
    else
      n->sibling = (BVH_Node*) ((size_t) n->sibling + (size_t) buf);
    n++;
  }
}

__global__ void globalBVH_Skippointer(int partition_no, int tree_height, int totalNodes)
{
  BVH_Node* root = deviceBVHRoot[partition_no];
  root->sibling = root+totalNodes;
  root->sibling->level = root->level;

  for(int i=0; i<tree_height-1; i++)
  {
    BVH_Node* node = root;
    for(int j=0; j<totalNodes; j++, node++)
    {
      if( (node->level == i) && (node->sibling == 0x0))
      {
        BVH_Node* skipNode = node->parent;

        if( skipNode->sibling != 0x0)
        {
          skipNode = skipNode->sibling;
        }
        else
        {
          while( (skipNode->sibling == 0x0) && (skipNode->level < tree_height-1/*root level*/))
          {
            skipNode = skipNode->parent;
          }

          if( skipNode->sibling != 0x0)  
            skipNode = skipNode->sibling;
        }
        node->sibling = skipNode; //node's sibling will be skip pointer
      }
    }
  }
}



/*
void BVH_RecursiveSearch()
{
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  float elapsed_time;

  //Open query file
  char queryFileName[100];

  if ( !strcmp(DATATYPE, "high"))
    sprintf(queryFileName, "/home/jwkim/inputFile/%s_dim_query.%d.bin.%ss", DATATYPE,NUMDIMS,SELECTIVITY);
  else
    sprintf(queryFileName, "/home/jwkim/inputFile/%s_dim_query.%d.bin.%ss.%s", DATATYPE,NUMDIMS,SELECTIVITY,querySize);

  FILE *fp = fopen(queryFileName, "r");
  if(fp==0x0){
    printf("Line %d : query file open error\n",__LINE__);
    printf("%s\n",queryFileName);
    exit(1);
  }


  int h_hit[NUMBLOCK];
  int h_count[NUMBLOCK];
  int h_rootCount[NUMBLOCK];

  long t_hit = 0;
  int t_count = 0;
  int t_rootCount = 0;

  int* d_hit;
  int* d_count;
  int* d_rootCount;

  cudaMalloc( (void**) &d_hit,   NUMBLOCK*sizeof(int) );
  cudaMalloc( (void**) &d_count, NUMBLOCK*sizeof(int) );
  cudaMalloc( (void**) &d_rootCount, NUMBLOCK*sizeof(int) );


  struct Rect* d_query;
  cudaMalloc( (void**) &d_query, NUMBLOCK*sizeof(struct Rect) );
  struct Rect query[NUMBLOCK];


  cudaEventRecord(start_event, 0);
  for( int i = 0 ; i < NUMSEARCH; ){
    int nBatch=0;
    for(nBatch=0; nBatch < NUMBLOCK && i < NUMSEARCH; nBatch++, i++) {
      for(int j=0;j<2*NUMDIMS;j++){
        fread(&query[nBatch].boundary[j], sizeof(float), 1, fp);
      }  
    }
    cudaMemcpy(d_query, query, nBatch*sizeof(struct Rect), cudaMemcpyHostToDevice);

    //FOR DEBUGGING
    globalBVHTreeSearchRecursive<<<1, 1>>>(d_query, d_hit, nBatch, PARTITIONED);

    cudaMemcpy(h_hit, d_hit, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_count, d_count, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rootCount, d_rootCount, nBatch*sizeof(int), cudaMemcpyDeviceToHost);

    for(int j=0;j<nBatch;j++){
      t_hit += h_hit[j];
      t_count += h_count[j];
      t_rootCount += h_rootCount[j]; 
    }
  }

  cudaThreadSynchronize();
  cudaEventRecord(stop_event, 0) ;
  cudaEventSynchronize(stop_event) ;
  cudaFree( d_query );


  cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
  printf("Recursive time          : %.3f ms\n", elapsed_time);
  printf("Recursive HIT           : %lu\n",t_hit);
  cout<< "Recursive visited       : " << t_count << endl;
  cout<< "Recursive root visited  : " << t_rootCount << endl;

  //selection ratio
#ifdef MPI
  printf("Recursive HIT ratio(%)  : %5.5f\n",((t_hit/mpiSEARCH)/(float)(NUMDATA/100)));
#else
  printf("Recursive HIT ratio(%)  : %5.5f\n",((t_hit/NUMSEARCH)/(float)(NUMDATA/100)));
#endif


  fclose(fp);

  cudaFree( d_hit );
  cudaFree( d_count );
  cudaFree( d_rootCount );


}
*/


__device__ int devBVHTreeSearch(BVH_Node *n, struct Rect *r)
{
  int hitCount = 0;
  int i;
  //printf("morton code %d\n",n->branch[0].hIndex);
  //printf("# of childs %d\n",n->count);
  //printf("level %d\n",n->level);

  if (n->level > 0) // this is an internal node in the tree
  {
    for (i=0; i<n->count; i++)
      if (devRectOverlap(r,&n->branch[i].rect))
      {
        hitCount += devBVHTreeSearch(n->branch[i].child, r );
      }
  }
  else // this is a leaf node 
  {
    /*
    for (i=0; i<n->count; i++)
      if (devRectOverlap(r,&n->branch[i].rect))
      {
        hitCount++;
      }
      */
    if( n->parent->level == n->level+1)
      hitCount++;
  }
  return hitCount;
}


/*
__global__ void globalBVHTreeSearchRecursive(struct Rect* query, int* hit, int mpiSEARCH, int PARTITIONED)
{

  int bid = blockIdx.x;
  //int tid = threadIdx.x;

  hit[bid] = 0;

  BVH_Node* root = deviceBVHRoot[0];

  int block_init_position = bid;
  int block_incremental_value = devNUMBLOCK;

  for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
    hit[bid] += devBVHTreeSearch(root, &query[n]);


}
*/

__device__ void device_print_BVHnode(BVH_Node* n )
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if( tid == 0 && bid == 0) {
    //DEBUG
    //printf(" # %d || %d  ",i,node);
    printf("node %lu\n",n);
    printf("morton code hIndex %lu\n",n->branch[n->count-1].mortonCode);
    printf("count  %d\n", n->count);
    printf("level  %d\n", n->level);

    if( n->level > 0)
    {
    
      for( int j=0; j<n->count; j++) {
        for( int d = 0; d < NUMDIMS*2 ; d ++) {
          printf(" dims : %d rect %5.5f \n",d , n->branch[j].rect.boundary[d]);
        }
        
        printf("%d child %lu\n", j, n->branch[j].child);
      }
    }
    printf("sibling %lu\n", n->sibling);
    printf("parent %lu\n",n->parent);
  
    printf("\n\n");
  }
}
__global__ void global_print_BVH(int partition_no, int tNODE)
{
  BVH_Node* node = deviceBVHRoot[partition_no];

  for( int i = 0 ; i < tNODE  ; i++){
    device_print_BVHnode(node);
    printf("\n\n");
    node++;
  }

}

