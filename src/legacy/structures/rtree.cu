
#include <rtree.h>

//#####################################################################
//######################## IMPLEMENTATION #############################
//#####################################################################


void InsertDataRTrees(bitmask_t* keys, struct Branch* data)
{

	char dataFileName[100];

	if( !strcmp(DATATYPE, "high" ) )
		sprintf(dataFileName, "/home/jwkim/inputFile/%s_dim_data.%d.bin\0", DATATYPE, NUMDIMS); 
	else
		sprintf(dataFileName, "/home/jwkim/inputFile/NOAA%d.bin",myrank); 

	//sprintf(dataFileName, "/tmp/ramdisk0/NOAA.bin"  ); 

	FILE *fp = fopen(dataFileName, "r");
	if(fp==0x0){
		printf("Line %d : Insert file open error\n", __LINE__);
		printf("%s\n",dataFileName);
		exit(1);
	}

	//!!! nBits*NUMDIMS can not equal and larger than 64(bitmask_t)

	struct Branch b;
	for(int i=0; i<NUMDATA; i++){
		bitmask_t coord[NUMDIMS];
		for(int j=0;j<NUMDIMS;j++){
			fread(&b.rect.boundary[j], sizeof(float), 1, fp);
			b.rect.boundary[NUMDIMS+j] = b.rect.boundary[j];
			//int tmp = b.rect.boundary[j]*1000000;
			//b.rect.boundary[NUMDIMS+j] = b.rect.boundary[j] = tmp/1000000.0f;
		}

		for(int j=0;j<NUMDIMS;j++){
			coord[j] = (bitmask_t) (1000000*b.rect.boundary[j]);
			//coord[j] = (bitmask_t) (1000000* ( ((float)(b.rect.boundary[j])+(float)(b.rect.boundary[j+NUMDIMS])) /(float)2.0));
		}


		/*
		for(int j=0;j<NUMDIMS;j++){
			fread(&b.rect.boundary[j], sizeof(float), 1, fp);
			b.rect.boundary[NUMDIMS+j] = b.rect.boundary[j];
			
			//same... why?
			//if( !strcmp(DATATYPE, "high"))
			//	coord[j] = (bitmask_t)(1000000*b.rect.boundary[j]);
			//else
			coord[j] = (bitmask_t)(1000000*b.rect.boundary[j]);
		}
		*/

		//synthetic
		if( !strcmp(DATATYPE, "high")){

			// sort the index with only 3 dim for highdimensional data
			//b.hIndex = hilbert_c2i(3, 20, coord);

			//for rect data
			//b.hIndex = hilbert_c2i(2, 20, coord);
			
			if( NUMDIMS == 2 )
				b.hIndex = hilbert_c2i(2, 31, coord);
			else
				b.hIndex = hilbert_c2i(3, 21, coord);
			
			/*
			bitmask_t nBits = 64/NUMDIMS; //64 is maximum dimensions
			if( NUMDIMS > nBits )
				b.hIndex = hilbert_c2i(NUMDIMS-1, nBits, coord);
			else{
				b.hIndex = hilbert_c2i(NUMDIMS, nBits-1, coord);
			}
			*/
		}
		else //real datasets from NOAA
			b.hIndex = hilbert_c2i(3, 20, coord);

		keys[i] = b.hIndex;
		data[i] = b;

		/*
		 * HilbertCurve BUG..
		bitmask_t coords[NUMDIMS];
		hilbert_i2c(NUMDIMS, 20, b.hIndex, coords);;

		for(int dd=0; dd<NUMDIMS; dd++)
		{
			if( b.rect.boundary[dd] != ((float)coords[dd]/(float)1000000.0f))
			{
				printf("rect[%d] : %f\n", dd, b.rect.boundary[dd]);
				printf("cv rect[%d] : %f\n", dd, (float)coords[dd]/(float)1000000.0f);
			}
		}
		*/
	}

	if(fclose(fp) != 0){
		printf("Line %d : Insert file close error\n", __LINE__);
		exit(1);
	}
}


// Hibert Comparing Function
int HilbertCompare(const void* a, const void* b)
{
	struct Branch* b1 = (struct Branch*) a;
	struct Branch* b2 = (struct Branch*) b;

	return (b1->hIndex > b2->hIndex) ? 1 : 0;
}

__global__ void globalReassignhilbertIndex(struct Branch* data, int NUMDATA )
{
	int tid = ( blockIdx.x *blockDim.x ) + threadIdx.x;

	while( tid < NUMDATA ){
		data[tid].hIndex = tid+1;
		tid+=524288;
	}
}


bool TreeDumpToFile()
{
	/*
	int tree_idx = 0;
	for( int partition_no = 0; partition_no < PARTITIONED; partition_no++)
	{
		if( partition_no == boundary_of_trees )
			tree_idx = 1;


		char indexFileName[100];
		sprintf(indexFileName,"/tmp/MPHR_Tree%ddims.%dfanout.%dndata.%dmpirank.%dpartitioned.idx", NUMDIMS, NODECARD, NUMDATA, myrank, partition_no);
		FILE *fp = fopen(indexFileName,"w+");

		if( fp == 0x0){
			//load indexed file if there exist
			printf("ERROR : FAILED TO CREATE AN INDEX FILE!!!\n");
			return false;
		}


		int nodeSeq = 0;
		nodeSeq++;

		int numberOfnode = indexSize[tree_idx]/PGSIZE; // total number of node in the this tree..


#if BUILD_ON_CPU == 0
		char* d_treeRoot;
		cudaMalloc((void**)&d_treeRoot, indexSize[tree_idx]);
		globalLoadFromMem<<<1,1>>>(d_treeRoot, partition_no, indexSize[tree_idx]/PGSIZE);
		cudaMemcpy(&ch_root[partition_no], d_treeRoot, indexSize[tree_idx], cudaMemcpyDeviceToHost);
#endif

		struct Node* node = (struct Node*)ch_root[partition_no];

		for( int i=0; i < numberOfnode; i ++ ){

			long off = ftell(fp);
			fwrite(node, PGSIZE, 1, fp);

			if(node->level > 0)
			{ // This is an internal node in the tree
				for(int j=0; j< node->count; j++)
				{
					long branch_off = nodeSeq * PGSIZE;
					nodeSeq++;

					fseek(fp, off + (j+1)*sizeof( struct Branch ) , SEEK_SET);
					fwrite(&branch_off, sizeof(long), 1, fp);
					fseek(fp, 0, SEEK_END);
				}
			}
			node++;
		}
		fclose(fp);
		cudaFree(d_treeRoot);
	}

	*/
	return true;
}


// returns true if index file is loaded successfully
bool TreeLoadFromFile()
{
	int tree_idx = 0;
	for( int partition_no = 0; partition_no < PARTITIONED; partition_no++)
	{
		if( partition_no == boundary_of_trees )
			tree_idx = 1;

		char indexFileName[100];
		sprintf(indexFileName,"/tmp/MPHR_Tree%ddims.%dfanout.%dndata.%dmpirank.%dpartitioned.idx", NUMDIMS, NODECARD, NUMDATA, myrank, partition_no);

		FILE *fp = fopen(indexFileName,"r");
		if( fp == 0x0){
			printf("ERROR : FAILED TO LOAD INDEX FILE!!!\n");
			//at least one index file does not exist, returns false
			return false;
		}

		//pageable memory
		//ch_root[partition_no] = (char*) malloc ( indexSize[tree_idx] );
		//pinned memory 
		checkCuda( cudaMallocHost( (void**)&ch_root[partition_no], indexSize[tree_idx]));

		int nodeSeq = 0;
		int numberOfnode = indexSize[tree_idx]/PGSIZE; // total number of node in the this tree..

		struct Node temp;

		while( nodeSeq < numberOfnode)
		{
			fread(&temp, PGSIZE, 1, fp);
			memcpy( ch_root[partition_no]+(PGSIZE*nodeSeq), &temp, PGSIZE );
			nodeSeq++;
		}


		char* d_treeRoot;
		cudaMalloc( (void**) & d_treeRoot, indexSize[tree_idx]);
		cudaMemcpy(d_treeRoot, &ch_root[partition_no], indexSize[tree_idx], cudaMemcpyHostToDevice);
		globalTreeLoadFromMem<<<1,1>>>(d_treeRoot, partition_no, NUMBLOCK, PARTITIONED, numberOfnode);
		cudaThreadSynchronize();


		fseek(fp, 0, SEEK_SET);
		nodeSeq = 0;
		while( nodeSeq < numberOfnode)
		{
			fread(&temp, PGSIZE, 1, fp);
			memcpy( ch_root[partition_no]+(PGSIZE*nodeSeq), &temp, PGSIZE );
			nodeSeq++;
		}

		struct Node* hostRoot = (struct Node*) ch_root[partition_no];
		//print_index( hostRoot , numberOfnode );
		TreeLoadFromMem(hostRoot, ch_root[partition_no] );

		fclose(fp);
		cudaFree(d_treeRoot);
	}

	printf("SUCCESS LOAD INDEX FROM FILE\n");

	return true;
} 


void TreeLoadFromMem(struct Node* n, char *buf)
{
	int i;

	if (n->level > 0) // this is an internal node in the tree 
	{
		//printf("node's level %d \n", n->level);
		for (i=0; i<n->count; i++) {
			// translate file offset to memory pointer
			n->branch[i].child = (struct Node*) ((size_t) n->branch[i].child + (size_t) buf);
			TreeLoadFromMem(n->branch[i].child, buf);
		}
	}

	return;
}

__global__ void globalLoadFromMem(char* d_treeRoot, int partition_no, int tNODE)
{
	struct Node* node = (struct Node*)d_treeRoot;
	struct Node* node2 = (struct Node*)deviceRoot[partition_no];

	for(int i=0; i<tNODE; i++, node++, node2++)
	{
		node->count = node2->count;
		node->level = node2->level;
		for(int j=0; j<node2->count; j++)
		{
			node->branch[j] = node2->branch[j];
		}
	}
}

__global__ void globalTreeLoadFromMem(char *buf, int partition_no, int NUMBLOCK, int PARTITIONED /* number of partitioned index*/, int numOfnodes)
{
	if( partition_no == 0 ){
		devNUMBLOCK = NUMBLOCK;
		deviceRoot = (struct Node**) malloc( sizeof(struct Node*) * PARTITIONED );
	}

	deviceRoot[partition_no] = (struct Node*) buf;
	devTreeLoadFromMem(deviceRoot[partition_no], buf, numOfnodes);
}

__device__ void devTreeLoadFromMem(struct Node* n, char *buf, int numOfnodes)
{

	for( int i = 0; i < numOfnodes; i ++){
		if( n->level > 0 ){
			for( int c = 0 ; c < n->count; c++){
				n->branch[c].child = (struct Node*) ((size_t) n->branch[c].child + (size_t) buf);
			}
		}
		n++;
	}
}

__device__ void device_print_Node_SOA(struct Node_SOA* n )
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	if( tid == 0 && bid == 0)
	{
		//DEBUG
		//printf(" # %d || %d  ",i,node);
		printf("node %lu\n",n);
		printf("morton code hIndex %lu\n",n->index[n->count-1]);
		printf("count  %d\n", n->count);
		printf("level  %d\n", n->level);

		if( n->level > 0)
		{
		
			for( int j=0; j<n->count; j++)
			{
				/*
				for( int d = 0; d < NUMDIMS*2 ; d ++)
				{
					printf(" dims : %d rect %5.5f \n",d , n->boundary[d*NODECARD+j]);
				}
				*/
				printf("%d child %lu\n", j, n->child[j]);
			}
		}
		printf("\n\n");
	}
}
__global__ void global_print_Node_SOA(int partition_no, int tNODE)
{
	struct Node_SOA* node = (struct Node_SOA*) deviceRoot[partition_no];

	for( int i = 0 ; i < tNODE  ; i++){
		device_print_Node_SOA(node);
		printf("\n\n");
		node++;
	}

}



void Build_Rtrees(int m)
{

//	printf("####################################################\n");
//	printf("################# INSERT DATA ######################\n");
//	printf("####################################################\n");

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	//Insert data and calculate the Hilbert value 
	bitmask_t keys[NUMDATA];
	struct Branch data[NUMDATA];
	cudaEventRecord(start_event, 0);

	InsertDataRTrees(keys, data);

	cudaEventRecord(stop_event, 0) ;
	cudaEventSynchronize(stop_event) ;
	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	printf("Insertion time = %.3fs\n\n", elapsed_time/1000.0f);


//	printf("####################################################\n");
//	printf("############# SORT THE DATA ON CPU #################\n");
//	printf("####################################################\n");

	/*
	//To compare with GPU

	struct Branch data2[NUMDATA];
	memcpy( data2, data, sizeof(struct Branch)*NUMDATA);

	cudaEventRecord(start_event, 0);

	//!!! NEED TO BE PARALLEL
	qsort(data2, NUMDATA, sizeof(struct Branch), HilbertCompare);


	//reassign on CPU
	for( int i = 0; i < NUMDATA; i++)
		data2[i].hIndex = i+1;

	//Figure out the percentage of duplication data
	
//	int cnt=0;
//	for(int i=0; i<NUMDATA-1; i++)
//	{
//		if( data2[i].hIndex == data2[i+1].hIndex)
//			cnt++;
//			//printf("%d data %lu\n",i, data[i].hIndex);
//	}
//	printf("Duplication hilbert index is %d ( %5.2f %) \n",cnt, (cnt/(float)NUMDATA)*100);

	cudaEventRecord(stop_event, 0) ;
	cudaEventSynchronize(stop_event) ;
	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
	printf("Sort Time on CPU = %.3fs\n", elapsed_time/1000.0f);
	*/


//	printf("####################################################\n");
//	printf("############# SORT THE DATA ON GPU #################\n");
//	printf("####################################################\n");

	cudaEventRecord(start_event, 0);

	//copy host to device
	thrust::device_vector<bitmask_t> d_keys(NUMDATA);
	thrust::device_vector<struct Branch> d_data(NUMDATA);
	thrust::copy(keys, keys+NUMDATA, d_keys.begin() );
	thrust::copy(data, data+NUMDATA, d_data.begin() );

	//sort on GPU
	thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_data.begin() );

	//ReassignHilbertINdex on GPU
	struct Branch* t_data = thrust::raw_pointer_cast(&d_data[0]);
	globalReassignhilbertIndex<<<1024,512>>>( t_data , NUMDATA);



	thrust::copy(d_data.begin(), d_data.end(), data );

	cudaEventRecord(stop_event, 0) ;
	cudaEventSynchronize(stop_event) ;
	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
	printf("Sort Time on GPU = %.3fs\n\n", elapsed_time/1000.0f);

	/*
	for(int i=0; i<NUMDATA; i++)
	{
		for(int d=0; d<NUMDIMS; d++)
		{
		printf("node[%d], index %llu, min rect[%d] %f\n",i, data[i].hIndex , d, data[i].rect.boundary[d]);
		printf("node[%d], index %llu, max rect[%d] %f\n",i, data[i].hIndex, d+NUMDIMS, data[i].rect.boundary[d+NUMDIMS]);
		}
	}
	*/

	//Distributed policy 
	if( POLICY ){
		struct Branch temp[NUMDATA];
		memcpy( temp, data, sizeof(struct Branch)*NUMDATA);

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

//	printf("####################################################\n");
//	printf("############### BUILD THE INDEX ####################\n");
//	printf("####################################################\n");

	if( m == 1 && METHOD[5] == true)
	{
		Bulk_loading_with_parentLink(data);
		int tree_idx = 0;
		for (int partition_no = 0; partition_no < PARTITIONED; partition_no++)
		{
			if( partition_no == boundary_of_trees )
				tree_idx = 1;
			globaltranspose_BVHnode<<<1,NODECARD>>>(partition_no,indexSize[tree_idx]/BVH_PGSIZE );
		}
	}
	else if( m == 2 &&  METHOD[6] == true)
	{
		Bulk_loading_with_skippointer(data);
		int tree_idx = 0;
		for (int partition_no = 0; partition_no < PARTITIONED; partition_no++)
		{
			if( partition_no == boundary_of_trees )
				tree_idx = 1;
			globaltranspose_BVHnode<<<1,NODECARD>>>(partition_no,(indexSize[tree_idx]/BVH_PGSIZE)+1 );
		}
	}
	else
	{
		Bulk_loading(data);
		int tree_idx = 0;
		for (int partition_no = 0; partition_no < PARTITIONED; partition_no++)
		{
			if( partition_no == boundary_of_trees )
				tree_idx = 1;
			globaltranspose_node<<<1,NODECARD>>>(partition_no,indexSize[tree_idx]/PGSIZE );
		}
	}

	size_t avail, total;
	cudaMemGetInfo( &avail, &total );
	size_t used = total-avail;
	//DEBUG
	printf(" Used %lu / Total %lu ( %.2f % ) \n\n\n",used,total, ( (double)used/(double)total)*100);


}


void Bulk_loading( struct Branch* data )
{
	cudaEvent_t start_event, stop_event;
	checkCuda ( cudaEventCreate(&start_event) );
	checkCuda ( cudaEventCreate(&stop_event) );

	//measure memory transfer time btw host and device
	cudaEvent_t memcpy_start, memcpy_stop;
	checkCuda ( cudaEventCreate(&memcpy_start) );
	checkCuda ( cudaEventCreate(&memcpy_stop) );

	float elapsed_time;
	float memcpy_elapsed_time;

	//###########################################################################
	//################### ATTACH THE LEAF NODE ON THE INDEX #####################
	//###########################################################################

	checkCuda ( cudaEventRecord(start_event, 0) );

	boundary_of_trees = NUMDATA%PARTITIONED; // the boundary number which that distinguish trees by tree height

	// there are trees which have different tree heights
	int partitioned_NUMDATA[2];
	partitioned_NUMDATA[0] = partitioned_NUMDATA[1] = (NUMDATA/PARTITIONED);
	printf("partitioned_NUMDATA %d\n", partitioned_NUMDATA[0]);
	printf("partitioned_NUMDATA %d\n", partitioned_NUMDATA[1]);
	if( boundary_of_trees )
		partitioned_NUMDATA[0]++;

	// if number of data is not exactly divide by number of block,
	tree_height[0] = ceil ( log( partitioned_NUMDATA[0] ) / log( NODECARD ) ) ;
	tree_height[1] = ceil ( log( partitioned_NUMDATA[1] ) / log( NODECARD ) );

	//DEBUGGING
	printf("Tree height 0 is %d\n",tree_height[0]);
	printf("Tree height 1 is %d\n",tree_height[1]);


	//int number_of_node_in_level[2][ tree_height[0] ] ;
	number_of_node_in_level[0] = (int*)malloc(sizeof(int)*tree_height[0]);
	number_of_node_in_level[1] = (int*)malloc(sizeof(int)*tree_height[0]);

	number_of_node_in_level[0][0] = ceil ( (double) partitioned_NUMDATA[0]/NODECARD  )  ;
	number_of_node_in_level[1][0] = ceil ( (double) partitioned_NUMDATA[1]/NODECARD  )  ;

	indexSize[0] = indexSize[1] = 0;

	for ( int i = 0 ; i < tree_height[1]-1 ; i ++ )
	{
		number_of_node_in_level[0][i+1] = ceil ( (double)number_of_node_in_level[0][i]/NODECARD ) ;
		number_of_node_in_level[1][i+1] = ceil ( (double)number_of_node_in_level[1][i]/NODECARD ) ;
	}


	if( tree_height[0] > tree_height[1] ){
		number_of_node_in_level[0][tree_height[0]-1] = ceil ( (double)number_of_node_in_level[0][tree_height[0]-2]/NODECARD )  ;
		number_of_node_in_level[0][tree_height[0]-1] = 0;
	}

	for ( int i = tree_height[1]-1 ; i >= 0 ; i -- )
	{
		printf("level : %d number of node : %d \n",i, number_of_node_in_level[0][i]);
		printf("level : %d number of node : %d \n",i, number_of_node_in_level[1][i]);
		indexSize[0] += number_of_node_in_level[0][i]*PGSIZE;
		indexSize[1] += number_of_node_in_level[1][i]*PGSIZE;
	}
	if( tree_height[0] > tree_height[1] )
		indexSize[0] += number_of_node_in_level[0][tree_height[0]-1]*PGSIZE;
	int partitioned_offset = 0 ;

	int tree_idx = 0;
	for ( int partition_no = 0 ; partition_no < PARTITIONED; partition_no++)
	{
		if( partition_no == boundary_of_trees )
			tree_idx = 1;

		//pageable memory
		//ch_root[partition_no] = (char*) malloc ( indexSize[tree_idx] );
		//cd_root[partition_no] = (char*) malloc ( indexSize[tree_idx] );
		//pinned memory
		checkCuda( cudaMallocHost( (void**)& ch_root[partition_no] , indexSize[tree_idx] ));
		checkCuda( cudaMallocHost( (void**)& cd_root[partition_no] , indexSize[tree_idx] ));

		//attach the leafnodes to the memory buffer
		int count_and_level[2] = { NODECARD /* count */ , 0 /* level */ };
		long offset = indexSize[tree_idx] - ( number_of_node_in_level[tree_idx][0] * PGSIZE ) ;

		for( int i = 0 ; i < number_of_node_in_level[tree_idx][0]-1 ; i ++ )
		{
			memcpy(ch_root[partition_no]+offset,   count_and_level,   sizeof(int)*2 );
			memcpy(ch_root[partition_no]+offset+8, data + NODECARD*i + partitioned_offset, sizeof(struct Branch)*NODECARD);
			offset += PGSIZE ;
		}
		//last node in leaf level, change the count if necessary
		if( partitioned_NUMDATA[tree_idx] % NODECARD )
			count_and_level[0] = partitioned_NUMDATA[tree_idx] % NODECARD; // count
		memcpy(ch_root[partition_no]+offset, count_and_level, sizeof(int)*2 );
		memcpy(ch_root[partition_no]+offset+8, data + NODECARD * (number_of_node_in_level[tree_idx][0]-1) + partitioned_offset , sizeof(struct Branch)*count_and_level[0]); 

		memcpy(cd_root[partition_no], ch_root[partition_no], indexSize[tree_idx]);

		partitioned_offset += partitioned_NUMDATA[tree_idx];

		//DEBUGGING
		//print leaves for debugging
		//offset = indexSize[tree_idx] - ( number_of_node_in_level[tree_idx][0] * PGSIZE ) ;
		//struct Node* temp = (struct Node*)(ch_root[partition_no]+offset);
		//printf("PRINT LEAVES for DEBUGGING\n");
		//print_index(temp, number_of_node_in_level[tree_idx][0]);

	} 


	checkCuda ( cudaEventRecord(stop_event, 0) );
	checkCuda ( cudaEventSynchronize(stop_event) );
	checkCuda ( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
	printf("Attaching Time = %.3fs\n\n", elapsed_time/1000.0f);


	//###########################################################################
	//######################## BUILD-UP USING CPU ###############################
	//###########################################################################


	tree_idx = 0;
	checkCuda ( cudaEventRecord(start_event, 0) );
	for ( int partition_no = 0; partition_no < PARTITIONED ; partition_no ++ )
	{
		if( partition_no == boundary_of_trees)
			tree_idx = 1;

		char* root = ch_root[partition_no];
		struct Node* ptr_node, *parent_node ;
		int offset = indexSize[tree_idx];

		for( int level = 0 ; level < tree_height[tree_idx]-1 ; level ++ )
		{
			//start offset of each level
			offset -=  number_of_node_in_level[tree_idx][level] * PGSIZE ;

			ptr_node    = (struct Node*) ( (char*)root + offset ) ;
			parent_node = (struct Node*) ( (char*)root + ( offset - ( number_of_node_in_level[tree_idx][level+1] * PGSIZE ))) ;

			for( int n = 0 ; n < number_of_node_in_level[tree_idx][level] ; n ++ , ptr_node++)
			{
				//move to next parent node
				if( !( n % NODECARD ) ){
					if( n ) parent_node++;
					parent_node->level = level+1;
					parent_node->count = 0;
				}

				//set the child pointer of parent node to this node
				parent_node->branch[parent_node->count].child = ptr_node;

				//Find out the min, max boundaries in this node and set up the parent rect.
				float MIN_boundary = FLT_MAX;
				float MAX_boundary = FLT_MIN;
				for( int d = 0 ; d < NUMDIMS ; d ++ )
				{
					MIN_boundary = FLT_MAX;
					MAX_boundary = FLT_MIN;

					for( int c = 0 ; c < ptr_node->count ; c ++ )
					{
						if( MIN_boundary > ptr_node->branch[c].rect.boundary[d] )
							MIN_boundary = ptr_node->branch[c].rect.boundary[d];
						if( MAX_boundary < ptr_node->branch[c].rect.boundary[d+NUMDIMS] )
							MAX_boundary = ptr_node->branch[c].rect.boundary[d+NUMDIMS];
					}
					parent_node->branch[parent_node->count].rect.boundary[d]
						= MIN_boundary;
					parent_node->branch[parent_node->count].rect.boundary[d+NUMDIMS]
						= MAX_boundary;
				}

				//set the parent branch hilbert index
				parent_node->branch[parent_node->count].hIndex = ptr_node->branch[ptr_node->count-1].hIndex;
				//increaes count of parent node
				parent_node->count++;

			}
		}
		/*
		float area = .0f;
		struct Node* tmpRoot = (struct Node*)root;
		int overlapCnt = 0;
		for(int i1 = 0; i1<tmpRoot->count-1; i1++)
		{
			for(int i2 = i1+1; i2<tmpRoot->count; i2++)
			{
				if( RectOverlap(&tmpRoot->branch[i1].rect, &tmpRoot->branch[i2].rect) )
				{
					area += IntersectedRectArea(&tmpRoot->branch[i1].rect, &tmpRoot->branch[i2].rect);
					overlapCnt++;
				}
			}
		}
		printf("root node count %d overlap cnt %d Intersected area is %5.5f\n", tmpRoot->count, overlapCnt, area);
		*/
	}

	checkCuda ( cudaEventRecord(stop_event, 0) );
	checkCuda ( cudaEventSynchronize(stop_event) );
	checkCuda ( cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
	printf("Build Time on CPU = %.3fs\n\n", elapsed_time/1000.0f);

	//###########################################################################
	//######################## BUILD-UP USING GPU ###############################
	//###########################################################################


	long offset[2];
	long parent_offset[2];
	int number_of_node[2];

	long *d_offset;
	long *d_parent_offset;
	int *d_number_of_node;
	checkCuda ( cudaMalloc ((void**) &d_offset, sizeof(long)*2) );
	checkCuda ( cudaMalloc ((void**) &d_parent_offset, sizeof(long)*2) );
	checkCuda ( cudaMalloc ((void**) &d_number_of_node, sizeof(int)*2) );

	tree_idx = 0;

	offset[0] = (indexSize[0] / PGSIZE);
	offset[1] = (indexSize[1] / PGSIZE);


	char *d_treeRoot;

	//Measure the memcpy transfer time
	checkCuda( cudaEventRecord(memcpy_start, 0) );
	for (int partition_no = 0; partition_no < PARTITIONED; partition_no++){
		if (partition_no == boundary_of_trees)
			tree_idx = 1;
		cudaMalloc((void**) &d_treeRoot, indexSize[tree_idx]);
		checkCuda ( cudaMemcpy(d_treeRoot, cd_root[partition_no], indexSize[tree_idx], cudaMemcpyHostToDevice) );
		globalSetDeviceRoot<<<1,1>>>(d_treeRoot, partition_no, NUMBLOCK, PARTITIONED);
	}
	checkCuda ( cudaEventRecord(memcpy_stop, 0) );
	checkCuda ( cudaThreadSynchronize() );

	checkCuda( cudaEventRecord(start_event, 0) );

	for (int level  = 0; level < tree_height[0]-1; level++)
	{

		offset[0] -= number_of_node_in_level[0][level];
		offset[1] -= number_of_node_in_level[1][level];

		parent_offset[0] = offset[0] - number_of_node_in_level[0][level+1];
		parent_offset[1] = offset[1] - number_of_node_in_level[1][level+1];

		number_of_node[0] = number_of_node_in_level[0][level];
		number_of_node[1] = number_of_node_in_level[1][level];

		checkCuda ( cudaMemcpy(d_offset, offset, sizeof(long)*2, cudaMemcpyHostToDevice) );
		checkCuda ( cudaMemcpy(d_parent_offset, parent_offset, sizeof(long)*2, cudaMemcpyHostToDevice) );
		checkCuda ( cudaMemcpy(d_number_of_node, number_of_node, sizeof(int)*2, cudaMemcpyHostToDevice) );
//#ifndef ILP
//		globalBottomUpBuild<<<NUMBLOCK, NODECARD>>>(d_offset, d_parent_offset, d_number_of_node, boundary_of_trees, PARTITIONED);
//#else
		globalBottomUpBuild_ILP<<<NUMBLOCK, NUMTHREADS>>>(d_offset, d_parent_offset, d_number_of_node, boundary_of_trees, PARTITIONED);
//#endif
	}
	checkCuda ( cudaEventRecord(stop_event, 0) );
	checkCuda ( cudaEventSynchronize(stop_event) );

	checkCuda ( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
	checkCuda ( cudaEventElapsedTime(&memcpy_elapsed_time, memcpy_start, memcpy_stop));
	printf ("Build Time(Only Building) on GPU = %.3fs\n", elapsed_time/1000.0f);
	printf ("Build Time(Transfer Time) on GPU = %.3fs\n\n", memcpy_elapsed_time/1000.0f);

	//printf("Print index on the GPU\n");
	//tree_idx = 0;
	//for (int partition_no = 0; partition_no < PARTITIONED; partition_no++){
	//	if (partition_no == boundary_of_trees)
	//		tree_idx = 1;
	//	global_print_index<<<1,1>>>(partition_no, indexSize[tree_idx]/PGSIZE);
	//}


	cudaFree( d_offset );
	cudaFree( d_parent_offset );
	cudaFree( d_number_of_node );



}
void Bulk_loading_with_parentLink( struct Branch* data )
{
	cudaEvent_t start_event, stop_event;
	checkCuda ( cudaEventCreate(&start_event) );
	checkCuda ( cudaEventCreate(&stop_event) );

	//measure memory transfer time btw host and device
	cudaEvent_t memcpy_start, memcpy_stop;
	checkCuda ( cudaEventCreate(&memcpy_start) );
	checkCuda ( cudaEventCreate(&memcpy_stop) );

	float elapsed_time;
	float memcpy_elapsed_time;

	//###########################################################################
	//################### ATTACH THE LEAF NODE ON THE INDEX #####################
	//###########################################################################

	checkCuda ( cudaEventRecord(start_event, 0) );

	boundary_of_trees = NUMDATA%PARTITIONED; // the boundary number which that distinguish trees by tree height

	// there are trees which have different tree heights
	int partitioned_NUMDATA[2];
	partitioned_NUMDATA[0] = partitioned_NUMDATA[1] = (NUMDATA/PARTITIONED);
	if( boundary_of_trees )
		partitioned_NUMDATA[0]++;

	// if number of data is not exactly divide by number of block,
	tree_height[0] = ceil ( log( partitioned_NUMDATA[0] ) / log( NODECARD ) ) ;
	tree_height[1] = ceil ( log( partitioned_NUMDATA[1] ) / log( NODECARD ) );

	//DEBUGGING
	printf("Tree height 0 is %d\n",tree_height[0]);
	printf("Tree height 1 is %d\n",tree_height[1]);


	//int number_of_node_in_level[2][ tree_height[0] ] ;
	number_of_node_in_level[0] = (int*)malloc(sizeof(int)*tree_height[0]);
	number_of_node_in_level[1] = (int*)malloc(sizeof(int)*tree_height[0]);

	number_of_node_in_level[0][0] = ceil ( (double) partitioned_NUMDATA[0]/NODECARD  )  ;
	number_of_node_in_level[1][0] = ceil ( (double) partitioned_NUMDATA[1]/NODECARD  )  ;

	indexSize[0] = indexSize[1] = 0;

	for ( int i = 0 ; i < tree_height[1]-1 ; i ++ )
	{

		number_of_node_in_level[0][i+1] = ceil ( (double)number_of_node_in_level[0][i]/NODECARD ) ;
		number_of_node_in_level[1][i+1] = ceil ( (double)number_of_node_in_level[1][i]/NODECARD ) ;
	}
	if( tree_height[0] > tree_height[1] ){
		number_of_node_in_level[0][tree_height[0]-1] = ceil ( (double)number_of_node_in_level[0][tree_height[0]-2]/NODECARD )  ;
		number_of_node_in_level[0][tree_height[0]-1] = 0;
	}

	for ( int i = tree_height[1]-1 ; i >= 0 ; i -- )
	{
		printf("level : %d number of node : %d \n",i, number_of_node_in_level[0][i]);
		printf("level : %d number of node : %d \n",i, number_of_node_in_level[1][i]);
		indexSize[0] += number_of_node_in_level[0][i]*BVH_PGSIZE;
		indexSize[1] += number_of_node_in_level[1][i]*BVH_PGSIZE;
	}
	if( tree_height[0] > tree_height[1] )
		indexSize[0] += number_of_node_in_level[0][tree_height[0]-1]*BVH_PGSIZE;
	int partitioned_offset = 0 ;

	int tree_idx = 0;
	for ( int partition_no = 0 ; partition_no < PARTITIONED; partition_no++)
	{
		if( partition_no == boundary_of_trees )
			tree_idx = 1;

		//pageable memory
		//ch_root[partition_no] = (char*) malloc ( indexSize[tree_idx] );
		//cd_root[partition_no] = (char*) malloc ( indexSize[tree_idx] );
		//pinned memory
		checkCuda( cudaMallocHost( (void**)& ch_root[partition_no] , indexSize[tree_idx] ));
		checkCuda( cudaMallocHost( (void**)& cd_root[partition_no] , indexSize[tree_idx] ));

		//attach the leafnodes to the memory buffer
		int count_and_level[2] = { NODECARD /* count */ , 0 /* level */ };
		long offset = indexSize[tree_idx] - ( number_of_node_in_level[tree_idx][0] * BVH_PGSIZE ) ;

		for( int i = 0 ; i < number_of_node_in_level[tree_idx][0]-1 ; i ++ )
		{
			memcpy(ch_root[partition_no]+offset,   count_and_level,   sizeof(int)*2 );
			memcpy(ch_root[partition_no]+offset+8, data + NODECARD*i + partitioned_offset, sizeof(struct Branch)*NODECARD);
			offset += BVH_PGSIZE ;
		}

		//last node in leaf level, change the count if necessary
		if( partitioned_NUMDATA[tree_idx] % NODECARD )
			count_and_level[0] = partitioned_NUMDATA[tree_idx] % NODECARD; // count
		memcpy(ch_root[partition_no]+offset, count_and_level, sizeof(int)*2 );
		memcpy(ch_root[partition_no]+offset+8, data + NODECARD * (number_of_node_in_level[tree_idx][0]-1) + partitioned_offset , sizeof(BVH_Branch)*count_and_level[0]); 

		memcpy(cd_root[partition_no], ch_root[partition_no], indexSize[tree_idx]);

		partitioned_offset += partitioned_NUMDATA[tree_idx];

		//DEBUGGING
		//print leaves for debugging
		//offset = indexSize[tree_idx] - ( number_of_node_in_level[tree_idx][0] * PGSIZE ) ;
		//struct Node* temp = (struct Node*)(ch_root[partition_no]+offset);
		//printf("PRINT LEAVES for DEBUGGING\n");
		//print_index(temp, number_of_node_in_level[tree_idx][0]);

	} 


	checkCuda ( cudaEventRecord(stop_event, 0) );
	checkCuda ( cudaEventSynchronize(stop_event) );
	checkCuda ( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
	printf("Attaching Time = %.3fs\n\n", elapsed_time/1000.0f);


	//###########################################################################
	//######################## BUILD-UP USING CPU ###############################
	//###########################################################################

	//###########################################################################
	//######################## BUILD-UP USING GPU ###############################
	//###########################################################################


	long offset[2];
	long parent_offset[2];
	int number_of_node[2];

	long *d_offset;
	long *d_parent_offset;
	int *d_number_of_node;
	checkCuda ( cudaMalloc ((void**) &d_offset, sizeof(long)*2) );
	checkCuda ( cudaMalloc ((void**) &d_parent_offset, sizeof(long)*2) );
	checkCuda ( cudaMalloc ((void**) &d_number_of_node, sizeof(int)*2) );

	tree_idx = 0;

	offset[0] = (indexSize[0] / BVH_PGSIZE);
	offset[1] = (indexSize[1] / BVH_PGSIZE);


	char *d_treeRoot;

	//Measure the memcpy transfer time
	checkCuda( cudaEventRecord(memcpy_start, 0) );
	for (int partition_no = 0; partition_no < PARTITIONED; partition_no++){
		if (partition_no == boundary_of_trees)
			tree_idx = 1;
		cudaMalloc((void**) &d_treeRoot, indexSize[tree_idx]);
		checkCuda ( cudaMemcpy(d_treeRoot, cd_root[partition_no], indexSize[tree_idx], cudaMemcpyHostToDevice) );
		globalSetDeviceBVHRoot<<<1,1>>>(d_treeRoot, partition_no, NUMBLOCK, PARTITIONED);
	}
	checkCuda ( cudaEventRecord(memcpy_stop, 0) );
	checkCuda ( cudaThreadSynchronize() );

	checkCuda( cudaEventRecord(start_event, 0) );

	for (int level  = 0; level < tree_height[0]-1; level++){

		offset[0] -= number_of_node_in_level[0][level];
		offset[1] -= number_of_node_in_level[1][level];

		parent_offset[0] = offset[0] - number_of_node_in_level[0][level+1];
		parent_offset[1] = offset[1] - number_of_node_in_level[1][level+1];

		number_of_node[0] = number_of_node_in_level[0][level];
		number_of_node[1] = number_of_node_in_level[1][level];

		checkCuda ( cudaMemcpy(d_offset, offset, sizeof(long)*2, cudaMemcpyHostToDevice) );
		checkCuda ( cudaMemcpy(d_parent_offset, parent_offset, sizeof(long)*2, cudaMemcpyHostToDevice) );
		checkCuda ( cudaMemcpy(d_number_of_node, number_of_node, sizeof(int)*2, cudaMemcpyHostToDevice) );
//#ifndef ILP
//		globalBottomUpBuild_with_parentLink<<<NUMBLOCK, NODECARD>>>(d_offset, d_parent_offset, d_number_of_node, boundary_of_trees, PARTITIONED);
//#else
		globalBottomUpBuild_ILP_with_parentLink<<<NUMBLOCK, NUMTHREADS>>>(d_offset, d_parent_offset, d_number_of_node, boundary_of_trees, PARTITIONED);
//#endif
	}
	checkCuda ( cudaEventRecord(stop_event, 0) );
	checkCuda ( cudaEventSynchronize(stop_event) );

	checkCuda ( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
	checkCuda ( cudaEventElapsedTime(&memcpy_elapsed_time, memcpy_start, memcpy_stop));
	//printf ("Build Time(Only Building) on GPU = %.3fs\n", elapsed_time/1000.0f);
	//printf ("Build Time(Transfer Time) on GPU = %.3fs\n\n", memcpy_elapsed_time/1000.0f);

	//printf("Print index on the GPU\n");
	//tree_idx = 0;
	//for (int partition_no = 0; partition_no < PARTITIONED; partition_no++){
	//	if (partition_no == boundary_of_trees)
	//		tree_idx = 1;
	//	global_print_index<<<1,1>>>(partition_no, indexSize[tree_idx]/PGSIZE);
	//}


	cudaFree( d_offset );
	cudaFree( d_parent_offset );
	cudaFree( d_number_of_node );
	//size_t avail, total;
	//cudaMemGetInfo( &avail, &total );
	//size_t used = total-avail;
	//DEBUG
	//printf(" Used %lu / Total %lu ( %.2f % ) \n\n\n",used,total, ( (double)used/(double)total)*100);


}

void Bulk_loading_with_skippointer( struct Branch* data )
{
	cudaEvent_t start_event, stop_event;
	checkCuda ( cudaEventCreate(&start_event) );
	checkCuda ( cudaEventCreate(&stop_event) );

	//measure memory transfer time btw host and device
	cudaEvent_t memcpy_start, memcpy_stop;
	checkCuda ( cudaEventCreate(&memcpy_start) );
	checkCuda ( cudaEventCreate(&memcpy_stop) );

	float elapsed_time;
	//###########################################################################
	//################### ATTACH THE LEAF NODE ON THE INDEX #####################
	//###########################################################################

	checkCuda ( cudaEventRecord(start_event, 0) );

	boundary_of_trees = NUMDATA%PARTITIONED; // the boundary number which that distinguish trees by tree height

	// there are trees which have different tree heights
	int partitioned_NUMDATA[2];
	partitioned_NUMDATA[0] = partitioned_NUMDATA[1] = (NUMDATA/PARTITIONED);
	if( boundary_of_trees )
		partitioned_NUMDATA[0]++;

	// if number of data is not exactly divide by number of block,
	tree_height[0] = ceil ( log( partitioned_NUMDATA[0] ) / log( NODECARD ) ) ;
	tree_height[1] = ceil ( log( partitioned_NUMDATA[1] ) / log( NODECARD ) );

	//DEBUGGING
	printf("Tree height 0 is %d\n",tree_height[0]);
	printf("Tree height 1 is %d\n",tree_height[1]);


	//int number_of_node_in_level[2][ tree_height[0] ] ;
	number_of_node_in_level[0] = (int*)malloc(sizeof(int)*tree_height[0]);
	number_of_node_in_level[1] = (int*)malloc(sizeof(int)*tree_height[0]);

	number_of_node_in_level[0][0] = ceil ( (double) partitioned_NUMDATA[0]/NODECARD  )  ;
	number_of_node_in_level[1][0] = ceil ( (double) partitioned_NUMDATA[1]/NODECARD  )  ;

	indexSize[0] = indexSize[1] = 0;

	for ( int i = 0 ; i < tree_height[1]-1 ; i ++ )
	{
		number_of_node_in_level[0][i+1] = ceil ( (double)number_of_node_in_level[0][i]/NODECARD ) ;
		number_of_node_in_level[1][i+1] = ceil ( (double)number_of_node_in_level[1][i]/NODECARD ) ;
	}
	if( tree_height[0] > tree_height[1] ){
		number_of_node_in_level[0][tree_height[0]-1] = ceil ( (double)number_of_node_in_level[0][tree_height[0]-2]/NODECARD )  ;
		number_of_node_in_level[0][tree_height[0]-1] = 0;
	}

	for ( int i = tree_height[1]-1 ; i >= 0 ; i -- )
	{
		//printf("level : %d number of node : %d \n",i, number_of_node_in_level[partition_no][i]);
		indexSize[0] += number_of_node_in_level[0][i]*BVH_PGSIZE;
		indexSize[1] += number_of_node_in_level[1][i]*BVH_PGSIZE;
	}
	if( tree_height[0] > tree_height[1] )
		indexSize[0] += number_of_node_in_level[0][tree_height[0]-1]*BVH_PGSIZE;

	//DEBUGGING
	printf("Number of nodes is %d\n",(indexSize[0]/BVH_PGSIZE));
	printf("Number of nodes is %d\n",(indexSize[1]/BVH_PGSIZE));


	int partitioned_offset = 0 ;

	int tree_idx = 0;
	for ( int partition_no = 0 ; partition_no < PARTITIONED; partition_no++)
	{
		if( partition_no == boundary_of_trees )
			tree_idx = 1;

		//pageable memory
		//ch_root[partition_no] = (char*) malloc ( indexSize[tree_idx] );
		//cd_root[partition_no] = (char*) malloc ( indexSize[tree_idx] );
		//pinned memory
		checkCuda( cudaMallocHost( (void**)& ch_root[partition_no] , indexSize[tree_idx] ));
		checkCuda( cudaMallocHost( (void**)& cd_root[partition_no] , indexSize[tree_idx] ));

		//attach the leafnodes to the memory buffer
		int count_and_level[2] = { NODECARD /* count */ , 0 /* level */ };
		unsigned long long offset = indexSize[tree_idx] - ( number_of_node_in_level[tree_idx][0] * BVH_PGSIZE ) ;

		for( int i = 0 ; i < number_of_node_in_level[tree_idx][0]-1 ; i ++ )
		{
			memcpy(ch_root[partition_no]+offset,   count_and_level,   sizeof(int)*2 );
			memcpy(ch_root[partition_no]+offset+8, data + NODECARD*i + partitioned_offset, sizeof(struct Branch)*NODECARD);
			offset += BVH_PGSIZE ;
		}

		//last node in leaf level, change the count if necessary
		if( partitioned_NUMDATA[tree_idx] % NODECARD )
			count_and_level[0] = partitioned_NUMDATA[tree_idx] % NODECARD; // count
		memcpy(ch_root[partition_no]+offset, count_and_level, sizeof(int)*2 );
		memcpy(ch_root[partition_no]+offset+8, data + NODECARD * (number_of_node_in_level[tree_idx][0]-1) + partitioned_offset , sizeof(struct Branch)*count_and_level[0]); 

		memcpy(cd_root[partition_no], ch_root[partition_no], indexSize[tree_idx]);
		partitioned_offset += partitioned_NUMDATA[tree_idx];


	} 


	checkCuda ( cudaEventRecord(stop_event, 0) );
	checkCuda ( cudaEventSynchronize(stop_event) );
	checkCuda ( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
	printf("Attaching Time = %.3fs\n\n", elapsed_time/1000.0f);




	//###########################################################################
	//######################## BUILD-UP USING CPU ###############################
	//###########################################################################


	tree_idx = 0;
	checkCuda ( cudaEventRecord(start_event, 0) );
	for ( int partition_no = 0; partition_no < PARTITIONED ; partition_no++ )
	{
		if( partition_no == boundary_of_trees)
			tree_idx = 1;

		char* root = ch_root[partition_no];
		BVH_Node* ptr_node, *parent_node ;
		long offset = indexSize[tree_idx];

		for( int level = 0 ; level < tree_height[tree_idx]-1 ; level ++ )
		{
			//start offset of each level
			offset -=  number_of_node_in_level[tree_idx][level] * BVH_PGSIZE ;

			ptr_node    = (BVH_Node*) ( (char*)root + offset ) ;
			parent_node = (BVH_Node*) ( (char*)root + ( offset - ( number_of_node_in_level[tree_idx][level+1] * BVH_PGSIZE ))) ;

			for( int n = 0 ; n < number_of_node_in_level[tree_idx][level] ; n ++ , ptr_node++)
			{
				//move to next parent node
				if( !( n % NODECARD ) ){
					if( n ) parent_node++;
					parent_node->level = level+1;
					parent_node->count = 0;
				}


				//set the child pointer of parent node to this node
				parent_node->branch[parent_node->count].child = ptr_node;
				ptr_node->parent = parent_node;

				//Find out the min, max boundaries in this node and set up the parent rect.
				float MIN_boundary = FLT_MAX;
				float MAX_boundary = FLT_MIN;
				for( int d = 0 ; d < NUMDIMS ; d ++ )
				{
					MIN_boundary = FLT_MAX;
					MAX_boundary = FLT_MIN;

					for( int c = 0 ; c < ptr_node->count ; c ++ )
					{
						if( MIN_boundary > ptr_node->branch[c].rect.boundary[d] )
							MIN_boundary = ptr_node->branch[c].rect.boundary[d];
						if( MAX_boundary < ptr_node->branch[c].rect.boundary[d+NUMDIMS] )
							MAX_boundary = ptr_node->branch[c].rect.boundary[d+NUMDIMS];
					}
					parent_node->branch[parent_node->count].rect.boundary[d]
						= MIN_boundary;
					parent_node->branch[parent_node->count].rect.boundary[d+NUMDIMS]
						= MAX_boundary;
				}

				//set the parent branch hilbert index
				parent_node->branch[parent_node->count].mortonCode = ptr_node->branch[ptr_node->count-1].mortonCode;
				//increaes count of parent node
				parent_node->count++;

			}
		}

//printf("%d\n",__LINE__);

		unsigned long long off = 0;
		char* buf = (char*)malloc(indexSize[tree_idx]);
		char* buf2 = (char*)malloc(indexSize[tree_idx]);

//printf("%d\n",__LINE__);
		BVHTreeDumpToMem( (BVH_Node*)root, buf, tree_height[tree_idx]);
//printf("%d\n",__LINE__);
//printf("off %lu\n",off);
		BVHTreeDumpToMemDFS( buf, buf2, 0, &off );
//printf("%d\n",__LINE__);
		LinkUpSibling2( buf2, (indexSize[tree_idx]/BVH_PGSIZE));
		memcpy(buf, buf2, indexSize[tree_idx]);

//printf("%d\n",__LINE__);
		char* d_treeRoot;
		cudaMalloc( (void**) & d_treeRoot, indexSize[tree_idx]+BVH_PGSIZE);
		cudaMemcpy(d_treeRoot, buf, indexSize[tree_idx], cudaMemcpyHostToDevice);

//printf("%d\n",__LINE__);
		globalBVHTreeLoadFromMem<<<1,1>>>(d_treeRoot, partition_no, NUMBLOCK, PARTITIONED, indexSize[tree_idx]/BVH_PGSIZE );
		globalBVH_Skippointer<<<1,1>>>(partition_no, tree_height[tree_idx], indexSize[tree_idx]/BVH_PGSIZE);
		//checkDumpedTree(root, indexSize[tree_idx]/BVH_PGSIZE );
		//global_print_BVH<<<1,1>>>(partition_no, indexSize[tree_idx]/BVH_PGSIZE);
//printf("%d\n",__LINE__);

	}

	
	checkCuda ( cudaEventRecord(stop_event, 0) );
	checkCuda ( cudaEventSynchronize(stop_event) );
	checkCuda ( cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
	//printf("Build Time on CPU = %.3fs\n\n", elapsed_time/1000.0f);

	//###########################################################################
	//######################## BUILD-UP USING GPU ###############################
	//###########################################################################


	//size_t avail, total;
	//cudaMemGetInfo( &avail, &total );
	//size_t used = total-avail;
	//DEBUG
	//printf(" Used %lu / Total %lu ( %.2f % ) \n\n\n",used,total, ( (double)used/(double)total)*100);

}



__global__ void globalBottomUpBuild(long *_offset, long *_parent_offset, int *_number_of_node, int boundary_index, int PARTITIONED )
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int tree_index;
	int root_index = 0;
	int block_incremental_value = devNUMBLOCK;
	int n = bid;

	// this is not DP
	if( PARTITIONED > 1 ){
		root_index = bid;
		block_incremental_value=1; //task parallelism
		n = 0;
	}

	if( boundary_index > bid )
		tree_index = 0;
	else
		tree_index = 1;

	long offset = _offset[tree_index];
	long parent_offset = _parent_offset[tree_index];
	int number_of_node = _number_of_node[tree_index];

	struct Node* root = deviceRoot[root_index];
	struct Node* ptr_node;
	struct Node* parent_node;


	while( n < number_of_node )
	{
		ptr_node		= root + offset 			 + n;
		parent_node = root + parent_offset + (int)(n/NODECARD);

		parent_node->branch[ ( n % NODECARD ) ].child = ptr_node;
		parent_node->branch[ ( n % NODECARD ) ].hIndex = ptr_node->branch[ptr_node->count-1].hIndex;

		parent_node->level = (ptr_node->level)+1;
		parent_node->count = NODECARD;

		//Find out the min, max boundaries in this node and set up the parent rect.
		for( int d = 0 ; d < NUMDIMS ; d ++ )
		{
			int hd = d+NUMDIMS;

			__shared__ float lower_boundary[NODECARD];
			__shared__ float upper_boundary[NODECARD];



			if( tid < ptr_node->count )
			{
				lower_boundary[ tid ] = ptr_node->branch[tid].rect.boundary[d];
				upper_boundary[ tid ] = ptr_node->branch[tid].rect.boundary[hd];
			}
			else
			{
				lower_boundary[ tid ] = 1.0f;
				upper_boundary[ tid ] = 0.0f;
			}


			if ( tid < NODECARD/2 ){
				int N = NODECARD/2 + NODECARD%2;
				while(N > 1){
					if ( tid < N){
						if(lower_boundary[tid] > lower_boundary[tid+N])
							lower_boundary[tid] = lower_boundary[tid+N];
					}
					N = N/2 + N%2;
					__syncthreads();
				}
				if(tid==0) {
					if(N==1) {
						if( lower_boundary[0] > lower_boundary[1])
							lower_boundary[0] = lower_boundary[1];
					}
				}
			}
			else
			{
				int _tid = tid-(NODECARD/2);
				int N = NODECARD/2 + NODECARD%2;
				while(N > 1){
					if ( _tid < N ){
						if(upper_boundary[_tid] < upper_boundary[_tid+N])
							upper_boundary[_tid] = upper_boundary[_tid+N];
					}
					N = N/2 + N%2;
					__syncthreads();
				}
				if(_tid==0) {
					if(N==1) {
						if ( upper_boundary[0] < upper_boundary[1] )
							upper_boundary[0] = upper_boundary[1];
					}
				}
			}

			if( tid == 0 ){
				parent_node->branch[ ( n % NODECARD ) ].rect.boundary[d] = lower_boundary[0];
				parent_node->branch[ ( n % NODECARD ) ].rect.boundary[hd] = upper_boundary[0];
			}

			__syncthreads();
		}

		n+=block_incremental_value;
	}

	//last node in the level
	if(  number_of_node % NODECARD ){
		parent_node = root + offset - 1;
		if( number_of_node < NODECARD ) {
			parent_node->count = number_of_node; 
		}else{
			parent_node->count = (number_of_node%NODECARD);
		}
	}
}
__global__ void globalBottomUpBuild_ILP(long *_offset, long *_parent_offset, int *_number_of_node, int boundary_index, int PARTITIONED )
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int tree_index;
	int root_index = 0;
	int block_incremental_value = devNUMBLOCK;
	int n = bid;

	// this is not DP
	if( PARTITIONED > 1 ){
		root_index = bid;
		block_incremental_value=1; //task parallelism
		n = 0;
	}

	if( boundary_index > bid )
		tree_index = 0;
	else
		tree_index = 1;

	long offset = _offset[tree_index];
	long parent_offset = _parent_offset[tree_index];
	int number_of_node = _number_of_node[tree_index];

	struct Node* root = deviceRoot[root_index];
	struct Node* ptr_node;
	struct Node* parent_node;


	while( n < number_of_node )
	{
		ptr_node		= root + offset 			 + n;
		parent_node = root + parent_offset + (int)(n/NODECARD);

		parent_node->branch[ ( n % NODECARD ) ].child = ptr_node;
		parent_node->branch[ ( n % NODECARD ) ].hIndex = ptr_node->branch[ptr_node->count-1].hIndex;

		parent_node->level = (ptr_node->level)+1;
		parent_node->count = NODECARD;

		//Find out the min, max boundaries in this node and set up the parent rect.
		for( int d = 0 ; d < NUMDIMS ; d ++ )
		{
			int hd = d+NUMDIMS;

			__shared__ float lower_boundary[NODECARD];
			__shared__ float upper_boundary[NODECARD];



			for ( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS)
			{
				if( thread < ptr_node->count )
				{
					lower_boundary[ thread ] = ptr_node->branch[thread].rect.boundary[d];
					upper_boundary[ thread ] = ptr_node->branch[thread].rect.boundary[hd];
				}
				else
				{
					lower_boundary[ thread ] = 1.0f;
					upper_boundary[ thread ] = 0.0f;
				}
			}

			//threads in half get lower boundary

			int N = NODECARD/2 + NODECARD%2;
			while(N > 1){
				for ( int thread = tid ; thread < N ; thread+=NUMTHREADS){
					if(lower_boundary[thread] > lower_boundary[thread+N])
						lower_boundary[thread] = lower_boundary[thread+N];
				}
				N = N/2 + N%2;
				__syncthreads();
			}
			if(tid==0) {
				if(N==1) {
					if( lower_boundary[0] > lower_boundary[1])
						lower_boundary[0] = lower_boundary[1];
				}
			}
			//other half threads get upper boundary
			N = NODECARD/2 + NODECARD%2;
			while(N > 1){
				for ( int thread = tid ; thread < N ; thread+=NUMTHREADS){
					if(upper_boundary[thread] < upper_boundary[thread+N])
						upper_boundary[thread] = upper_boundary[thread+N];
				}
				N = N/2 + N%2;
				__syncthreads();
			}
			if(tid==0) {
				if(N==1) {
					if ( upper_boundary[0] < upper_boundary[1] )
						upper_boundary[0] = upper_boundary[1];
				}
			}

			if( tid == 0 ){
				parent_node->branch[ ( n % NODECARD ) ].rect.boundary[d] = lower_boundary[0];
				parent_node->branch[ ( n % NODECARD ) ].rect.boundary[hd] = upper_boundary[0];
			}

			__syncthreads();
		}

		n+=block_incremental_value;
	}

	//last node in the level
	if(  number_of_node % NODECARD ){
		parent_node = root + offset - 1;
		if( number_of_node < NODECARD ) {
			parent_node->count = number_of_node; 
		}else{
			parent_node->count = (number_of_node%NODECARD);
		}
	}
}

__global__ void globalBottomUpBuild_with_parentLink(long *_offset, long *_parent_offset, int *_number_of_node, int boundary_index, int PARTITIONED )
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int tree_index;
	int root_index = 0;
	int block_incremental_value = devNUMBLOCK;
	int n = bid;

	// this is not DP
	if( PARTITIONED > 1 ){
		root_index = bid;
		block_incremental_value=1; //task parallelism
		n = 0;
	}

	if( boundary_index > bid )
		tree_index = 0;
	else
		tree_index = 1;

	long offset = _offset[tree_index];
	long parent_offset = _parent_offset[tree_index];
	int number_of_node = _number_of_node[tree_index];

	BVH_Node* root = deviceBVHRoot[root_index];
	BVH_Node* ptr_node;
	BVH_Node* parent_node;


	while( n < number_of_node )
	{
		ptr_node		= root + offset 			 + n;
		parent_node = root + parent_offset + (int)(n/NODECARD);

		parent_node->branch[ ( n % NODECARD ) ].child = ptr_node;
		parent_node->branch[ ( n % NODECARD ) ].mortonCode = ptr_node->branch[ptr_node->count-1].mortonCode;

		ptr_node->parent = parent_node;

		parent_node->level = (ptr_node->level)+1;
		parent_node->count = NODECARD;

		//Find out the min, max boundaries in this node and set up the parent rect.
		for( int d = 0 ; d < NUMDIMS ; d ++ )
		{
			int hd = d+NUMDIMS;

			__shared__ float lower_boundary[NODECARD];
			__shared__ float upper_boundary[NODECARD];



			if( tid < ptr_node->count )
			{
				lower_boundary[ tid ] = ptr_node->branch[tid].rect.boundary[d];
				upper_boundary[ tid ] = ptr_node->branch[tid].rect.boundary[hd];
			}
			else
			{
				lower_boundary[ tid ] = 1.0f;
				upper_boundary[ tid ] = 0.0f;
			}


			if ( tid < NODECARD/2 ){
				int N = NODECARD/2 + NODECARD%2;
				while(N > 1){
					if ( tid < N){
						if(lower_boundary[tid] > lower_boundary[tid+N])
							lower_boundary[tid] = lower_boundary[tid+N];
					}
					N = N/2 + N%2;
					__syncthreads();
				}
				if(tid==0) {
					if(N==1) {
						if( lower_boundary[0] > lower_boundary[1])
							lower_boundary[0] = lower_boundary[1];
					}
				}
			}
			else
			{
				int _tid = tid-(NODECARD/2);
				int N = NODECARD/2 + NODECARD%2;
				while(N > 1){
					if ( _tid < N ){
						if(upper_boundary[_tid] < upper_boundary[_tid+N])
							upper_boundary[_tid] = upper_boundary[_tid+N];
					}
					N = N/2 + N%2;
					__syncthreads();
				}
				if(_tid==0) {
					if(N==1) {
						if ( upper_boundary[0] < upper_boundary[1] )
							upper_boundary[0] = upper_boundary[1];
					}
				}
			}

			if( tid == 0 ){
				parent_node->branch[ ( n % NODECARD ) ].rect.boundary[d] = lower_boundary[0];
				parent_node->branch[ ( n % NODECARD ) ].rect.boundary[hd] = upper_boundary[0];
			}

			__syncthreads();
		}

		n+=block_incremental_value;
	}

	//last node in the level
	if(  number_of_node % NODECARD ){
		parent_node = root + offset - 1;
		if( number_of_node < NODECARD ) {
			parent_node->count = number_of_node; 
		}else{
			parent_node->count = (number_of_node%NODECARD);
		}
	}
}
__global__ void globalBottomUpBuild_ILP_with_parentLink(long *_offset, long *_parent_offset, int *_number_of_node, int boundary_index, int PARTITIONED )
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int tree_index;
	int root_index = 0;
	int block_incremental_value = devNUMBLOCK;
	int n = bid;

	// this is not DP
	if( PARTITIONED > 1 ){
		root_index = bid;
		block_incremental_value=1; //task parallelism
		n = 0;
	}

	if( boundary_index > bid )
		tree_index = 0;
	else
		tree_index = 1;

	long offset = _offset[tree_index];
	long parent_offset = _parent_offset[tree_index];
	int number_of_node = _number_of_node[tree_index];

	BVH_Node* root = deviceBVHRoot[root_index];
	BVH_Node* ptr_node;
	BVH_Node* parent_node;


	while( n < number_of_node )
	{
		ptr_node		= root + offset 			 + n;
		parent_node = root + parent_offset + (int)(n/NODECARD);

		if( ptr_node->level == 0 )
		{
			if( (n+1) == number_of_node )
				ptr_node->sibling = NULL;
			else
				ptr_node->sibling = ptr_node+1;
		}
		parent_node->branch[ ( n % NODECARD ) ].child = ptr_node;
		parent_node->branch[ ( n % NODECARD ) ].mortonCode = ptr_node->branch[ptr_node->count-1].mortonCode;

		parent_node->level = (ptr_node->level)+1;
		parent_node->count = NODECARD;

		ptr_node->parent = parent_node;

		//Find out the min, max boundaries in this node and set up the parent rect.
		for( int d = 0 ; d < NUMDIMS ; d ++ )
		{
			int hd = d+NUMDIMS;

			__shared__ float lower_boundary[NODECARD];
			__shared__ float upper_boundary[NODECARD];



			for ( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS)
			{
				if( thread < ptr_node->count )
				{
					lower_boundary[ thread ] = ptr_node->branch[thread].rect.boundary[d];
					upper_boundary[ thread ] = ptr_node->branch[thread].rect.boundary[hd];
				}
				else
				{
					lower_boundary[ thread ] = 1.0f;
					upper_boundary[ thread ] = 0.0f;
				}
			}

			//threads in half get lower boundary

			int N = NODECARD/2 + NODECARD%2;
			while(N > 1){
				for ( int thread = tid ; thread < N ; thread+=NUMTHREADS){
					if(lower_boundary[thread] > lower_boundary[thread+N])
						lower_boundary[thread] = lower_boundary[thread+N];
				}
				N = N/2 + N%2;
				__syncthreads();
			}
			if(tid==0) {
				if(N==1) {
					if( lower_boundary[0] > lower_boundary[1])
						lower_boundary[0] = lower_boundary[1];
				}
			}
			//other half threads get upper boundary
			N = NODECARD/2 + NODECARD%2;
			while(N > 1){
				for ( int thread = tid ; thread < N ; thread+=NUMTHREADS){
					if(upper_boundary[thread] < upper_boundary[thread+N])
						upper_boundary[thread] = upper_boundary[thread+N];
				}
				N = N/2 + N%2;
				__syncthreads();
			}
			if(tid==0) {
				if(N==1) {
					if ( upper_boundary[0] < upper_boundary[1] )
						upper_boundary[0] = upper_boundary[1];
				}
			}

			if( tid == 0 ){
				parent_node->branch[ ( n % NODECARD ) ].rect.boundary[d] = lower_boundary[0];
				parent_node->branch[ ( n % NODECARD ) ].rect.boundary[hd] = upper_boundary[0];
			}

			__syncthreads();
		}

		n+=block_incremental_value;
	}

	//last node in the level
	if(  number_of_node % NODECARD ){
		parent_node = root + offset - 1;
		if( number_of_node < NODECARD ) {
			parent_node->count = number_of_node; 
		}else{
			parent_node->count = (number_of_node%NODECARD);
		}
	}
}

