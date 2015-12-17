// Tree Traversal :: 1. store ractangle and traverse tree structure using MPRS
//                   2. calculate ractangle and traverse tree structure using MPRS
//                   3. shrink bounding box when traversing tree structure from root to leaf 
//                   4. traverse tree structure via comparing common prefix and given range query's hilbert index 
//
//
//
// Tree Structure :: 5. Auxiliary leaf nodes can be two different types, one is RadixTreeNode_SOA and another is RadixTreeLeafNode
//                      (without auxiliary leaf nodes, node utilization is almost 3 % )
//                   6. Leaf nodes are redundant(should be deleted from the memory) because we have RadixTreeNode_SOA or RadixTreeLeafNode
//                   9. Leaf node utilization is low... 
//
//                   8. Leaf nodes only use left child, so that thread can access data in sparsely 
//                   7. When the degree of tree is 128, child pointers can be 128 or 256,
//
//
// TO DO :: 1. To improve experiments, tree index can be dumped into the file
//
// Problems :: 1. It shows a different output when we convert multi-dimensional data into hilbert index, and vice versa. 
//             2. Some of data has the same hilbert index..
//             3. Internal nodes have bounding boxes that aren't minimum 
//             4. Link phase should be fast
//
//
// MAIN FOCUS :: 1. Compare the number of visited tree nodes between HilbertRadixTree and MPHR-tree
//               2. How much of space can be reduced by HilbertRadixTree
//

#include <radix.h>

int test_num = 0;
int test_cnt = 0;

int test_num2= 0;
int test_cnt2= 0;

int _DEBUG_ = 0;

//#####################################################################
//######################## IMPLEMENTATION #############################
//#####################################################################

//convert multi-dimensional data into single dimensional data
//by using hilbert space filling curve
void InsertData(bitmask_t* data, float* data_rect)
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

	struct Branch b;
	for(int i=0; i<NUMDATA; i++){
		bitmask_t coord[NUMDIMS];

		for(int j=0;j<NUMDIMS;j++){
			fread(&b.rect.boundary[j], sizeof(float), 1, fp);
			b.rect.boundary[NUMDIMS+j] = b.rect.boundary[j];
			data_rect[i*NUMDIMS+j] = b.rect.boundary[j];
			//printf("data[%d] %d, %f\n", i, j, b.rect.boundary[j]);
		}

		for(int j=0;j<NUMDIMS;j++){
			coord[j] = (bitmask_t) (1000000*b.rect.boundary[j]);
		}

		if( !strcmp(DATATYPE, "high")){
			if( NUMDIMS == 2 )
				b.hIndex = hilbert_c2i(2, 31, coord);
			else
				b.hIndex = hilbert_c2i(3, 20, coord);
		}
		else //real datasets from NOAA
			b.hIndex = hilbert_c2i(3, 20, coord);


		data[i] = b.hIndex;
		//printf("data[%d]  %llu\n", i, data[i]);
	}

	if(fclose(fp) != 0){
		printf("Line %d : Insert file close error\n", __LINE__);
		exit(1);
	}
}


void Build_RadixTrees(int index_type)
{
	//Measuring the execution time 
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
	float elapsed_time;

	//	printf("####################################################\n");
	//	printf("################# INSERT DATA ######################\n");
	//	printf("####################################################\n");

	bitmask_t *data = (bitmask_t*)malloc(sizeof(bitmask_t)*NUMDATA);
	if( data == NULL)
	{
		printf("DEBUG :: %d, alloc is failed \n",__LINE__);
		exit(1);
	}

	float *data_rect = (float*)malloc(sizeof(float)*NUMDIMS*NUMDATA); //temporary
	if( data_rect == NULL)
	{
		printf("DEBUG :: %d, alloc is failed \n",__LINE__);
		exit(1);
	}

	cudaEventRecord(start_event, 0);

	//convert multi-dimensional data into single dimensional data
	//by using hilbert space filling curve
	InsertData(data, data_rect);

	//for debug
	/*
	data[0] = (unsigned long long) 6;
	data[1] = (unsigned long long) 11;
	data[2] = (unsigned long long) 13;
	data[3] = (unsigned long long) 29;
	data[4] = (unsigned long long) 33;
	data[5] = (unsigned long long) 39;
	data[6] = (unsigned long long) 50;
	data[7] = (unsigned long long) 51;
	data[8] = (unsigned long long) 56;
	*/

	cudaEventRecord(stop_event, 0) ;
	cudaEventSynchronize(stop_event) ;
	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	printf("Insertion time = %.3fs\n\n", elapsed_time/1000.0f);

	//	printf("####################################################\n");
	//	printf("############# SORT THE DATA ON GPU #################\n");
	//	printf("####################################################\n");

	cudaEventRecord(start_event, 0);

	//Copy data from host to device
	thrust::device_vector<bitmask_t> d_data(NUMDATA);
	thrust::copy(data, data+NUMDATA, d_data.begin() );

	//Sort the data on the GPU
	thrust::stable_sort_by_key(d_data.begin(), d_data.end(), d_data.begin());

	thrust::copy(d_data.begin(), d_data.end(), data );

	cudaEventRecord(stop_event, 0) ;
	cudaEventSynchronize(stop_event) ;
	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
	printf("Sort Time on GPU = %.3fs\n\n", elapsed_time/1000.0f);
	thrust::device_vector<bitmask_t>().swap(d_data);


	//DEBUG
	/*
		 printf("sorted morton code ---\n");	
		 for(int i=0; i<NUMDATA-1; i++)
		 {
		 printf("%lu ", data[i].mortonCode);
		 if( i % 10 == 0 )
		 printf("\n");
	//if( data[i].mortonCode > data[i+1].mortonCode)
	//	printf("%lu ", data[i].mortonCode);
	}
	printf("sorted morton code ---\n");	
	*/

	//!!! distribution policy
	//low priority work
	/*
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
	*/

	//####################################################
	//############ BUILD UP   Radix Trees ##################
	//####################################################

	printf("Starting construction\n");

	int bound = ceil ( ( double) NUMDATA/PARTITIONED );
	for( int p=0; p<PARTITIONED; p++)
	{
		//leaf nodes are not contiguous in memory, we need auxiliary leaf nodes
		//that store data in sequential memory block, so that we can traverse tree
		//strucutre by parallel scanning
		struct RadixTreeNode_SOA *RadixTreeNodeArray;
	   RadixTreeNodeArray  = (struct RadixTreeNode_SOA*)malloc(sizeof(struct RadixTreeNode_SOA) * 2 * ((unsigned long long )ceil((double) NUMDATA/NODECARD) ));
		if( RadixTreeNodeArray == NULL)
		{
			printf("DEBUG :: %d, alloc is failed \n",__LINE__);
			exit(1);
		}
		unsigned long long array_index = CopyDataToLeafNodes(RadixTreeNodeArray, data, data_rect );

		//Building a HilbertRadix-tree by recursively partitioning given datasets
		unsigned long long numOfNode= 0;

		printf("DEBUG :: %d\n",__LINE__);
		ch_root[p] = (char *) generateRadixTreesHierarchy(data, RadixTreeNodeArray, bound*p /*first*/ , min( ( bound*(p+1))-1,NUMDATA-1) /*last*/, 1 /*level*/, &numOfNode );
		printf("DEBUG :: %d\n",__LINE__);
	
		//mapping internal nodes into NODE_SOA
		unsigned long long start_leaf_nodes;

		numOfNode = mappingTreeIntoArray((struct RadixTreeNode*) ch_root[p], RadixTreeNodeArray, data, data_rect, &start_leaf_nodes,array_index );
		printf("%llu tree nodes are used \n", numOfNode);

		printf("leaf nodes utilization %.2f\n", 0.01*test_num/test_cnt);
		printf("internal nodes utilization %.2f\n", 0.01*((float)(test_num2-test_num)/(float)(test_cnt2-test_num)));
		printf("total nodes utilization %.2f\n", 0.01*test_num2/test_cnt2);

		ch_root[p] = (char*) RadixTreeNodeArray;

		//ScanningHilbertRadixTree(RadixTreeNodeArray, start_leaf_nodes, numOfNode);

		//PrintRadixTree(RadixTreeNodeArray);
	

		//TraverseHilbertRadixTreeUsingStack(RadixTreeNodeArray );
		TraverseHilbertRadixTreeUsingMPHR( (struct RadixTreeNode_SOA*)ch_root[p]  );
		exit(1);

		// print RadixTreeNode SOA 
		/*
		printf("\n\n\n");
		for(int i=0; i<numOfNode; i++)
		{
			printf("Print Node %llu, count %d \n", &RadixTreeNodeArray[i], RadixTreeNodeArray[i].count);
			for(int d=0; d<RadixTreeNodeArray[i].count ; d++)
			{
//				printf("%dth child node : ", d);
//				printCommonPrefix( RadixTreeNodeArray[i].common_prefix[d], 0, 64, RadixTreeNodeArray[i].nbits[d]);
//				printf("\n\n");
//				for(int r=0; r<NUMDIMS; r++)
//				{
//					printf("\nleft min boundary [%d] %f \n", r, RadixTreeNodeArray[i].boundary[(2*NODECARD*r)+d]);
//					printf("\nleft max boundary [%d] %f \n", r, RadixTreeNodeArray[i].boundary[((NUMDIMS+r)*2*NODECARD)+d]);
//				}
				printf("\nnode level %d", RadixTreeNodeArray[i].level[d]);
				printf("\nnode left index %llu", RadixTreeNodeArray[i].index[d*2]);
				printf("\nnode right index %llu", RadixTreeNodeArray[i].index[d*2+1]);
				printf("\nleft child node %llu", RadixTreeNodeArray[i].child[d*2]);
				printf("\nright child node %llu \n\n", RadixTreeNodeArray[i].child[d*2+1]);
			}
		}
		*/

		unsigned long long indexSize = sizeof(struct RadixTreeNode_SOA)*numOfNode;
		char* buf = (char*)malloc(indexSize);

		if( buf == NULL)
		{
			printf("DEBUG :: %d, alloc is failed \n",__LINE__);
			exit(1);
		}

		printf("Dump RadixArray To Mem\n" );
		RadixArrayDumpToMem( (struct RadixTreeNode_SOA*)ch_root[p], buf, numOfNode); 

		/*
		printf("\n\n\n");
		for(int i=0; i<numOfNode; i++)
		{
			printf("Print Node %llu, count %d \n", &RadixTreeNodeArray[i], RadixTreeNodeArray[i].count);
			for(int d=0; d<RadixTreeNodeArray[i].count ; d++)
			{
				printf("\nnode level %d", RadixTreeNodeArray[i].level[d]);
				printf("\nnode left index %llu", RadixTreeNodeArray[i].index[d*2]);
				printf("\nnode right index %llu", RadixTreeNodeArray[i].index[d*2+1]);
				printf("\nleft child node %llu", RadixTreeNodeArray[i].child[d*2]);
				printf("\nright child node %llu \n\n", RadixTreeNodeArray[i].child[d*2+1]);
			}
		}
		*/

		//!!! TO DO : dump it to the file and read from that

		/*
		printf("\n\n\n print dumped radix array \n\n\n");
		for(int i=0; i<numOfNode; i++)
		{
			printf("Print Node %llu \n", &RadixTreeNodeArray[i]);
			for(int d=0; d<RadixTreeNodeArray[i].count ; d++)
			{
				printCommonPrefix( RadixTreeNodeArray[i].common_prefix[d], 0, 7, RadixTreeNodeArray[i].nbits[d]);
				printf("\nnode level %d", RadixTreeNodeArray[i].level[d]);
				printf("\nnode index %d", RadixTreeNodeArray[i].index[d]);
				printf("\nleft child node %llu", RadixTreeNodeArray[i].child[d*2]);
				printf("\nright child node %llu \n\n", RadixTreeNodeArray[i].child[d*2+1]);
			}
		}
		*/

		char* d_treeRoot;
		cudaMalloc( (void**) & d_treeRoot, indexSize); 
		cudaMemcpy(d_treeRoot, buf, indexSize, cudaMemcpyHostToDevice);

		globalRadixTreeLoadFromMem<<<1,1>>>(d_treeRoot, p, NUMBLOCK, PARTITIONED, numOfNode );

		/*
		printf("Print Radix Tree on the GPU\n");
		int* c;
		cudaMalloc( (void**) & c, sizeof(int)); 
		int h_c;
		int a,b;
		//scanf("%d %d", &a, &b);
		a=b=20;
		globalPrintRadixTree<<<1,1>>>(p, numOfNode, a, b, c);
		cudaMemcpy(&h_c, c, sizeof(int), cudaMemcpyDeviceToHost);
		printf("c %d \n",h_c);
		*/
		free(RadixTreeNodeArray);
	}

	size_t avail, total;
	cudaMemGetInfo( &avail, &total );
	size_t used = total-avail;
	//DEBUG
	printf(" Used %lu / Total %lu ( %.2f % ) \n\n\n",used,total, ( (double)used/(double)total)*100);

	//Need to debug
	//BVH_RecursiveSearch();
	free(data);
}


int findSplitAndCommonPrefix( bitmask_t *data, int first,int  last, int* nbits)
{
	// Identical Morton codes => split the range in the middle.
	//
	unsigned long long firstCode = data[first];
	unsigned long long lastCode = data[last];

	if (firstCode == lastCode)
		return (first + last) >> 1;


	// Calculate the number of highest bits that are the same
	// for all objects, using the count-leading-zeros intrinsic.

	int commonPrefix = __builtin_clzl(firstCode ^ lastCode);
	*nbits = commonPrefix;

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
			unsigned long long splitCode = data[newSplit];
			int splitPrefix = __builtin_clzl(firstCode ^ splitCode);

			if (splitPrefix > commonPrefix)
				split = newSplit; // accept proposal
		}
	}
	while (step > 1);

	return split;
}

unsigned long long leaf_index = 0;
unsigned long long soa_index = 0;

RadixTreeNode* generateRadixTreesHierarchy( bitmask_t* data, RadixTreeNode_SOA* RadixTreeNodeArray, int first, int last, int level, unsigned long long *numOfNode )
{
	printf("DEBUG :: %d\n",__LINE__);
	//if( last == first  )
	if( (last - first) <= NODECARD  )
	{
		RadixTreeNode* node = (RadixTreeNode*) malloc (sizeof(RadixTreeNode));	
		if( node == NULL)
		{
			printf("DEBUG :: %d, alloc is failed \n",__LINE__);
			exit(1);
		}

		node->common_prefix = data[first];
		node->nbits = 64;
		//node->index[0] = ++leaf_index;
		node->index[0] = first;
		//printf("new leaf node index %llu %llu \n", node->index[0], node->index[1]);
		//node->index = data[first];
		node->level = INT_MAX;
		node->child[0] = NULL;
		node->child[1] = NULL;

		(*numOfNode)++;

		while( RadixTreeNodeArray[soa_index].index[(RadixTreeNodeArray[soa_index].count*2)-1] < first)
		{
	   	printf("DEBUG :: %d\n",__LINE__);
	   	printf("soa index %d first %d  :: \n",soa_index, first);
	   	printf("%lu %lu  :: \n",RadixTreeNodeArray[soa_index].index[(RadixTreeNodeArray[soa_index].count*2)-1], first);
			soa_index++;
		}
		printf("Soa index %d\n", soa_index);
		node->child_soa = &RadixTreeNodeArray[soa_index];
		leaf_index = last;
		node->index[1] = leaf_index;

	   printf("DEBUG :: %d\n",__LINE__);
		return node;
	}
	printf("DEBUG :: %d\n",__LINE__);

	int nbits = 0;
	int split = findSplitAndCommonPrefix(data, first, last, &nbits);

	printf("DEBUG :: %d\n",__LINE__);

	RadixTreeNode* node = (RadixTreeNode*) malloc (sizeof(RadixTreeNode));	
	if( node == NULL)
	{
		printf("DEBUG :: %d, alloc is failed \n",__LINE__);
		exit(1);
	}

	printf("DEBUG :: %d\n",__LINE__);
	RadixTreeNode *leftChild, *rightChild;

	leftChild = generateRadixTreesHierarchy(data,RadixTreeNodeArray, first, split, level+1, numOfNode);
	if( leftChild->level == INT_MAX)
		leftChild->nbits = nbits;
	node->index[0] = leaf_index;
	printf("DEBUG :: %d\n",__LINE__);

	rightChild = generateRadixTreesHierarchy(data,RadixTreeNodeArray, split+1, last, level+1, numOfNode);
	if( rightChild->level == INT_MAX)
		rightChild->nbits = nbits;
	node->index[1] = leaf_index;
	printf("DEBUG :: %d\n",__LINE__);

	node->common_prefix = data[first]; 
	node->nbits = nbits;
	node->level = level;
	node->child[0] = leftChild;
	node->child[1] = rightChild;

	(*numOfNode)++;
	printf("DEBUG :: %d\n",__LINE__);

	return node;
}


void RadixArrayDumpToMem(RadixTreeNode_SOA* root,char* buf,  int numOfnodes)
{
	RadixTreeNode_SOA* node = root;
	unsigned long long off = 0;
	unsigned long long pgsize = sizeof(RadixTreeNode_SOA);


	for(int i=0; i<numOfnodes; i++)
	{
		for(int j=0; j<node->count*2; j++)
		{
				if( node->child[j] != NULL)
				{
					if( _DEBUG_)
					printf("_DEBUG_ %d || i %d j %d \n", __LINE__,i,j);

				  unsigned long long child_dist = (node->child[j] - root);
					child_dist *= sizeof(struct RadixTreeNode_SOA);

					if( _DEBUG_)
					printf("_DEBUG_ %d || child dist %llu\n", __LINE__, child_dist);

					unsigned long long * off2 = (unsigned long long *)node+(unsigned long long)j;

					if( _DEBUG_)
					printf("_DEBUG_ %d || node %llu, j %d off %llu \n", __LINE__, node, j, off2);

					memcpy(off2, &child_dist, 8);
				}
		}
		memcpy(buf+off, node, pgsize);
		off += pgsize;
		node++;
	}
}


int CopyDataToLeafNodes(struct RadixTreeNode_SOA* soa_array, bitmask_t* data, float* data_rect )
{
	// create auxiliary leaf nodes in sequential memory block to parallel process 
	printf("Create leaf nodes in sequential memory block to parallel process \n");
	unsigned long long g=0;
	//unsigned long long i=array_index;
	unsigned long long i=0;
	for(;g<NUMDATA; i++)
	{
		unsigned long long d;
		for(d=0;g<NUMDATA && d<NODECARD; d++ ){
			soa_array[i].common_prefix[d] = data[g];
			soa_array[i].index[d*2] = g+1;
			soa_array[i].index[d*2+1] = g+1;
			soa_array[i].nbits[d] = 64;
			soa_array[i].level[d] = AUXILIARY_LEAF_NODE;
			soa_array[i].child[d*2] = 0;
			soa_array[i].child[d*2+1] = 0;

			for(int dd=0; dd<NUMDIMS; dd++)
			{
				soa_array[i].boundary[(NODECARD*dd)+d] = data_rect[g*NUMDIMS+dd];
				soa_array[i].boundary[(NODECARD*(dd+NUMDIMS))+d] = data_rect[g*NUMDIMS+dd];
			}
			/*
			for(int dd=0; dd<NUMDIMS; dd++)
			{
				soa_array[i].boundary[(NODECARD*2*dd)+d] = data_rect[g*NUMDIMS+dd];
				soa_array[i].boundary[(NODECARD*2*(dd+NUMDIMS))+d] = data_rect[g*NUMDIMS+dd];

				soa_array[i].boundary[(NODECARD*(2*dd+1))+d] = data_rect[g*NUMDIMS+dd];
				soa_array[i].boundary[(NODECARD*(2*(dd+NUMDIMS)+1))+d] = data_rect[g*NUMDIMS+dd];
			}
			*/
			g++;

		}
		soa_array[i].count = d;
	}

	printf("Done\n");
	return i;
}


int mappingTreeIntoArray(struct RadixTreeNode* root, struct RadixTreeNode_SOA* soa_array, bitmask_t* data, float* data_rect, unsigned long long* start_leaf_nodes, unsigned long long _array_index )
{
	printf("2 level queue..\n");
	unsigned long long array_index = _array_index; // index to store current node into RadixTreeNode_SOA array

	//clear q1 and q2
	while(!radix_q1.empty())
		radix_q1.pop();
	while(!radix_q2.empty())
		radix_q2.pop();

	struct RadixTreeNode* node; // current node

	radix_q1.push(root);

	while(!radix_q1.empty())
	{
		radix_q2.push(radix_q1.front());
		radix_q1.pop();

		// find as much unskippable nodes as much NODECARD 
		// except when there is no more tree nodes to visit
		while( (radix_q2.size() < NODECARD) )
		{
			bool NomoreTreeNodes = true;

			node = radix_q2.front();	
			radix_q2.pop();

			// if current node is an internal nodes,
			// insert child node into the q1
			// and mark 'NomoreTreeNodes' as false,
			// which means there exist nodes to visit
			if( node->child[0] != NULL )
			{
				radix_q2.push(node->child[0]);
				NomoreTreeNodes = false;
				if( radix_q2.size() == NODECARD) break;
			}

			if( node->child[1] != NULL )
			{
				radix_q2.push(node->child[1]);
				NomoreTreeNodes = false;
			}else if( node->level == LEAF_NODE )
				radix_q2.push(node);

			// If we didin't insert any tree node in the q1, so current tree node is a leaf node,
			// then we need to know all tree nodes in q1 are leaf nodes
			// If so, we've got to quit finding another unskippable node
			if( NomoreTreeNodes == true)
			{
				int num_queue=radix_q2.size();
				int num_internal_nodes = 0;
				for(int i=0; i<num_queue; i++)
				{
					node = radix_q2.front();
					radix_q2.pop();
					if( node->level  != LEAF_NODE ) num_internal_nodes++;
					radix_q2.push(node);
				}
				// if there exists no more internal node
				if( !num_internal_nodes ) 
				{
					//printf("%d\n", radix_q2.size());
					test_num += radix_q2.size();
					test_cnt++;
					break;
				}
			}
		}
		test_num2 += radix_q2.size();
		test_cnt2++;

		//after finding unskippable nodes,
		//mappimg those tree nodes into an array

		int nums_q2= radix_q2.size();
		for(int i=0; i<nums_q2; i++)
		{

			node = radix_q2.front();
			//printf("pop a node %llu from q2\n", node->index[0]);
			radix_q3.push(node);
			radix_q2.pop();
		}

		for(int i=0; i<nums_q2; i++)
		{
			node = radix_q3.top();
			//printf("pop a node %llu from q3\n", node->index[0]);
			radix_q3.pop();

			soa_array[array_index].common_prefix[i] = node->common_prefix;
			soa_array[array_index].nbits[i] = node->nbits;
			soa_array[array_index].level[i] = node->level;
			soa_array[array_index].index[i*2] = node->index[0];
			soa_array[array_index].index[i*2+1] = node->index[1];

			if( node->child[0] != NULL )
			{
				radix_q1.push(node->child[0]);
				soa_array[array_index].child[i*2] = &soa_array[array_index+radix_q1.size()];
				//printf("%d, index %llu\n", __LINE__, soa_array[array_index].index[i*2]);
			}
			else
			{
				soa_array[array_index].child[i*2] = node->child_soa;
				//printf("%d, index %llu\n", __LINE__, soa_array[array_index].index[i*2]);
			}

			if( node->child[1] != NULL )
			{
				radix_q1.push(node->child[1]);
				soa_array[array_index].child[i*2+1] = &soa_array[array_index+radix_q1.size()];
				//printf("%d, index %llu\n", __LINE__, soa_array[array_index].index[i*2]);
			}
			else
			{
				soa_array[array_index].child[i*2+1] = node->child_soa;
				//printf("%d, index %llu\n", __LINE__, soa_array[array_index].index[i*2]);
			}

			//printCommonPrefix(node->common_prefix, 0, 7, node->nbits);
			//printf("\nnode level %d", node->level);
			//printf("\nnode index %d \n", node->index);
			//unskippablenodes.insert(node);

		}
		soa_array[array_index].count = nums_q2;
		array_index++;
	}

	// create auxiliary leaf nodes in sequential memory block to parallel process 
	/*
	printf("Create leaf nodes in sequential memory block to parallel process \n");
	unsigned long long g=0;
	unsigned long long i=array_index;
	for(;g<NUMDATA; i++)
	{
		unsigned long long d;
		for(d=0;g<NUMDATA && d<NODECARD; d++ ){
			soa_array[i].common_prefix[d] = data[g];
			soa_array[i].index[d*2] = g+1;
			soa_array[i].index[d*2+1] = g+1;
			soa_array[i].nbits[d] = 64;
			soa_array[i].level[d] = AUXILIARY_LEAF_NODE;
			soa_array[i].child[d*2] = 0;
			soa_array[i].child[d*2+1] = 0;

			for(int dd=0; dd<NUMDIMS; dd++)
			{
				soa_array[i].boundary[(NODECARD*dd)+d] = data_rect[g*NUMDIMS+dd];
				soa_array[i].boundary[(NODECARD*(dd+NUMDIMS))+d] = data_rect[g*NUMDIMS+dd];
			}
			g++;

		}
		soa_array[i].count = d;
	}
	*/
	unsigned long long numOfnodes = array_index;

	printf("Make bounding boxes for testing\n");

	//FOR TESTING 
	struct RadixTreeNode_SOA *node_soa = soa_array;

	//int temp = 0;
	for(int i=0; i<numOfnodes; i++, node_soa++)
	{
		//printf("node level 0 , count %d \n", node_soa->count);
		for(int d=0; d<node_soa->count; d++)
		{
			/*if( node_soa->level[d] == LEAF_NODE )// || node_soa->level[d] == 0) // 
			{
				//float rect[NUMDIMS*2];
				//ConvertHilbertIndexToBoundingBox(node_soa->common_prefix[d], 0, rect);

				bitmask_t coords[NUMDIMS];
				hilbert_i2c(NUMDIMS, 20, node_soa->common_prefix[d], coords);;

				for(int dd=0; dd<NUMDIMS; dd++)
				{
					node_soa->boundary[(NODECARD*2*dd)+d] = (float)coords[dd]/(float)1000000.0f;
					node_soa->boundary[(NODECARD*2*(dd+NUMDIMS))+d] = (float)coords[dd]/(float)1000000.0f;
					//printf("DEBUG [%d] %d %d\n",__LINE__, (NODECARD*2*dd)+d, (NODECARD*2*(dd+NUMDIMS))+d);
					node_soa->boundary[(NODECARD*(2*dd+1))+d] = (float)coords[dd]/(float)1000000.0f;
					node_soa->boundary[(NODECARD*(2*(dd+NUMDIMS)+1))+d] = (float)coords[dd]/(float)1000000.0f;
					//printf("DEBUG [%d] %d %d\n",__LINE__, (NODECARD*(2*dd+1))+d,(NODECARD*(2*(dd+NUMDIMS)+1))+d);

					//printf("node[%d], index %llu, min rect[%d] %f\n",temp,node_soa->common_prefix[d], dd, node_soa->boundary[(2*NODECARD*dd)+d]);
					//printf("node[%d], index %llu, max rect[%d] %f\n",temp, node_soa->common_prefix[d], dd+NUMDIMS, node_soa->boundary[(2*NODECARD*(dd+NUMDIMS))+d]);
				}
				//temp++;
			}
			else
			{
			*/
				float rect[NUMDIMS*2];
				unsigned long long prefix = node_soa->common_prefix[d];

				//printf("prefix %d \n", node_soa->nbits[d]);
				//printCommonPrefix(prefix, 0, 64, 64);
				//printf("\n\n");

				prefix = LeftShift( RightShift( prefix, 64-node_soa->nbits[d]), 64-node_soa->nbits[d]);

				ConvertHilbertIndexToBoundingBox(prefix, 64-node_soa->nbits[d], rect);

					//printf("prefix of left sub tree\n");
					//printCommonPrefix(prefix, 0, 64, 64);
					//printf("\n\n");

					for(int dd=0; dd<NUMDIMS; dd++)
					{
						//printf("min %d %f\n", dd, min[dd]);
						node_soa->boundary[(NODECARD*2*dd)+d] = rect[dd];
						node_soa->boundary[(NODECARD*2*(dd+NUMDIMS))+d] = rect[dd+NUMDIMS];
					}
					unsigned long long inc = LeftShift( 1, 63 - node_soa->nbits[d]); 
					prefix = prefix | inc;

					ConvertHilbertIndexToBoundingBox(prefix, 64 - node_soa->nbits[d], rect);
					//printf("prefix of right sub tree\n");
					//printCommonPrefix(prefix, 0, 64, 64);
					//printf("\n\n");

					for(int dd=0; dd<NUMDIMS; dd++)
					{
						//printf("max %d %f\n", dd, min[dd]);
						node_soa->boundary[(NODECARD*(2*dd+1))+d] = rect[dd];
						node_soa->boundary[(NODECARD*(2*(dd+NUMDIMS)+1))+d] = rect[dd+NUMDIMS];

						//node_soa->boundary[(2*NODECARD*dd)+d+NODECARD] = rect[dd];
						//node_soa->boundary[(2*NODECARD*(dd+NUMDIMS))+d+NODECARD] = rect[dd+NUMDIMS];
					}
			//}
		}
	}

	printf("Link both created leaf nodes and internal nodes\n");

	// link created leaf nodes and internal nodes or leaf node

	if( 0 )
   {
	for(int i=0; i<array_index; i++)
	{
		for(int dd=0; dd<soa_array[i].count;dd++)
		{
			//a leaf node
			if( soa_array[i].level[dd] == LEAF_NODE )
			{
				unsigned long long j;
				for(j=array_index; j<numOfnodes; j++)
				{
					//printf("DEBUG :: %d index %llu lastest index %llu \n",
					//__LINE__, soa_array[i].index[dd*2], soa_array[j].index[(soa_array[j].count*2)-1]);
					if( soa_array[i].index[dd*2] <= soa_array[j].index[(soa_array[j].count*2)-1])
					{
						soa_array[i].child[dd*2]   = &soa_array[j];
						/*
				    for(int d=0; d<soa_array[j].count; d++)
						{
							if( soa_array[i].common_prefix[dd] == soa_array[j].common_prefix[d])
							{
								soa_array[i].index[dd*2] = soa_array[j].index[dd*2];
								soa_array[i].index[dd*2+1] = soa_array[j].index[dd*2+1];
								break;
							}
						}
						*/
						break;
					}
				}
				if( j == numOfnodes )
					printf("DEBUG :: %d Something's wrong\n",__LINE__);
			}
			/*
			else // an internal nodes...
			{
				if( soa_array[i].child[dd*2] != NULL)
				{
					int ll=0;
					struct RadixTreeNode_SOA* child = soa_array[i].child[dd*2];
					for( int l=0; l< child->count; l++)
					{
						if( child->level[l] == LEAF_NODE )
							ll++;
					}
					if( ll == child->count ) // all the left child nodes of current node are leaf nodes
					{
						unsigned long long j;
						for(j=array_index; j<numOfnodes; j++)
						{
							if( child->index[0] <= soa_array[j].index[(soa_array[j].count*2)-1])
							{
								soa_array[i].child[dd*2] = &soa_array[j];
								break;
							}
						}
						if( j == numOfnodes )
							printf("DEBUG :: %d Something's wrong\n",__LINE__);
					}
				}
				if( soa_array[i].child[dd*2+1] != NULL)
				{
					int ll=0;
					struct RadixTreeNode_SOA* child = soa_array[i].child[dd*2+1];
					for( int l=0; l< child->count; l++)
					{
						if( child->level[l] == LEAF_NODE )
							ll++;
					}
					if( ll == child->count ) // all the right child nodes of current node are leaf nodes
					{
						unsigned long long j;
						for(j=array_index; j<numOfnodes; j++)
						{
							if( child->index[0] <= soa_array[j].index[(soa_array[j].count*2)-1])
							{
								soa_array[i].child[dd*2+1] = &soa_array[j];
								break;
							}
						}
						if( j == numOfnodes )
							printf("DEBUG :: %d Something's wrong\n",__LINE__);
					}
				}
			}
			*/
		}
	}//end for
	}


	printf("Mapping phase has been done!\n");

	*start_leaf_nodes = array_index;
	return numOfnodes;
}

//e.g.) traverseRadixTrees_BFS((RadixTreeNode*) ch_root[p] );
void traverseRadixTrees_BFS(struct RadixTreeNode* n) // Breadth-First Search
{
	while(!radix_q0.empty()){
		radix_q0.pop();
	}

	radix_q0.push(n);

	while(!radix_q0.empty()){
		struct RadixTreeNode* t = radix_q0.front();
		radix_q0.pop();

		printf("address %lu \n", t);
		printCommonPrefix(t->common_prefix, 0, 64, t->nbits);
		printf("\n");
		printf("The largest hilbert index %llu \n", t->index);
		printf("node level %d \n", t->level);
		printf("LeftChild %lu\n", t->child[0]);
		printf("RightChild %lu\n", t->child[1]);
		printf("\n\n");

		if( t->child[0] != NULL)
			radix_q0.push(t->child[0]);

		if( t->child[1] != NULL)
			radix_q0.push(t->child[1]);
	}
}


int TraverseRadixTree(struct RadixTreeNode_SOA* n, struct Rect *r )
{
	int hitCount = 0;
	int i;

	for (i=0; i<n->count*2; i++)
	{
		if( _DEBUG_)
		printf("i [%d] compare bounding box in an internal/leaf(%d) node %llu\n",i,n->level[i/2], n->index[i]);

		if ( n->level[i/2] != AUXILIARY_LEAF_NODE )
		{
			if( n->child[i] != NULL && RadixNode_SOA_Overlap(r, n, i) )
			{
				if( _DEBUG_)
					printf("select %d's child node %llu\n",i, n->child[i]);

					hitCount += TraverseRadixTree(n->child[i], r );
			}
		}
		else
		{
			if( _DEBUG_ )
			printf("Auxiliary leaf node, visits %llu ~ %llu node\n",n->index[0], n->index[n->count*2-1]);

			hitCount++;
			break;
			//if( i%2 == 0 && RadixNode_SOA_Overlap(r, n, i))// a leaf node 
			/*
			int temp = hitCount;
			if( RadixNode_SOA_Overlap(r, n, i))// a leaf node 
			{
				if( _DEBUG_)
					printf("leaf hit\n");
				hitCount++;
			}
			if( temp == hitCount)
				printf("Couldn't find any overlapped data in leaf\n");
				*/
		}
	}
	return hitCount;
}

int passed_index2=0;
/*
int TraverseRadixTree(struct RadixTreeNode_SOA* n, struct Rect *r, unsigned long long  index )
{
	int hitCount = 0;
	int i;

	if( n->level[0] != 0 ) // internal nodes
	{
		for (i=0; i<n->count*2; i++)
		{
			if( _DEBUG_ )
		  printf("i[%d] compare bounding box in an internal(level : %d) node index %llu, passed index %llu\n",i,n->level[i/2], n->index[i], passed_index2);
			if(n->child[i]!= NULL && RadixNode_SOA_Overlap(r, n, i))
			{
			  if( _DEBUG_ )
				printf("select %d's child node, pass index \n",i, n->index[i]);
				hitCount += TraverseRadixTree(n->child[i], r, n->index[i] );
			}
		}
	}
	else
	{
		for(i=0; i<n->count; i++)
		{
			if( n->index[i*2] > passed_index2 && RadixNode_SOA_Overlap(r, n, i*2) )
			{
			  hitCount++;
			}
		}
		passed_index2 = n->index[n->count*2-1];
	}
	return hitCount;
}
*/
void PrintRadixTree(struct RadixTreeNode_SOA* n) //without auxiliary leaf nodes
{
	int i;
	for (i=0; i<n->count*2; i++)
	{
			if ( n->child[i] != NULL && n->level[i/2] != LEAF_NODE ) // an internal nodes
			{
				PrintRadixTree(n->child[i] );
			}
			else // a leaf node
			{
				printf("[%d/%d] leaf node(level : %llu)  index %llu\n",i, n->count*2, n->level[i/2], n->index[i]);
			}
	}
}

/*
int leaf_index = 0;
void SettingIndexInRadixTree(struct RadixTreeNode_SOA* n) //without auxiliary leaf nodes
{
	int i;
	for (i=0; i<n->count*2; i++)
	{
			if ( n->child[i] != NULL ) // an internal nodes
			{
				SettingIndexInRadixTree(n->child[i] );
				n->index[i] = leaf_index;
			}
			else // a leaf node
			{
				n->index[i*2] = ++leaf_index;
				n->index[i*2+1] = n->index[i*2];

			}
	}
}

int  SettingIndexInRadixTree(struct RadixTreeNode_SOA* n)
{
	int maxIndex;
	int i;

	for (i=0; i<n->count*2; i++)
	{
		if ( n->child[i] != NULL )
		{
			maxIndex = SettingIndexInRadixTree(n->child[i] );
			if( n->level[i/2] != LEAF_NODE )
			  n->index[i] = maxIndex;
		}
	}
	return n->index[(n->count*2)-1];
}

*/
void ScanningHilbertRadixTree(struct RadixTreeNode_SOA* node, unsigned long long start_leaf_nodes, unsigned long long numOfnodes)
{
	//query
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
	}else
	{
		printf("Query Filename %s\n",queryFileName);
	}

	struct Rect query;

	int hit = 0;
	for( int s = 0 ; s < NUMSEARCH; s++ )
	{

		for(int d=0;d<NUMDIMS*2;d++)
			fread(&query.boundary[d], sizeof(float), 1, fp);

		for(unsigned long long i=start_leaf_nodes; i<numOfnodes; i++)
			for(int dd=0; dd<node[i].count; dd++)
				if( RadixNode_SOA_Overlap(&query, &node[i], dd*2))
					hit++;
	}
	printf("Scanning Hilbert Radix Tree\n Hit is %d\n", hit);
}

void TraverseHilbertRadixTreeUsingStack(struct RadixTreeNode_SOA* root)
{
	//query
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
	}else
	{
		printf("Query Filename %s\n",queryFileName);
	}

	struct Rect query;

	int hit = 0;
	for( int s = 0 ; s < NUMSEARCH; s++ )
	{
		passed_index2=0;
		for(int d=0;d<NUMDIMS*2;d++)
			fread(&query.boundary[d], sizeof(float), 1, fp);

if(_DEBUG_)
		printf("Start Search Operation with query [%d]\n", s);

		hit += TraverseRadixTree(root, &query);
		//printf("%dth search, hit is %d\n", s, hit);
	}
	printf("Traditional Tree Traversal in Hilbert Radix Tree\n Hit is %d\n", hit);
}
void TraverseHilbertRadixTreeUsingMPHR(RadixTreeNode_SOA* root)
{
	while( root->level[0] == AUXILIARY_LEAF_NODE )
	{
		printf("jump to the root node\n");
		root++;
	}
	//query
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
	}else
	{
		printf("Query Filename %s\n",queryFileName);
	}

	struct Rect query;

	int hit = 0;
	for( int s = 0 ; s < NUMSEARCH; s++ )
	{
		for(int d=0;d<NUMDIMS*2;d++)
			fread(&query.boundary[d], sizeof(float), 1, fp);

if(_DEBUG_)
		printf("Start Search Operation with query [%d]\n", s);

		unsigned long long passed_index = 0;
		unsigned long long last_index = root->index[(root->count*2)-1];
if(_DEBUG_)
		printf("Last index %llu\n", last_index);

		struct RadixTreeNode_SOA* node = root;

		while( passed_index < last_index )
		{	
if(_DEBUG_)
			printf("Passed index %llu  last index %llu\n", passed_index, last_index);

			while( node->level[0] != AUXILIARY_LEAF_NODE ) {
if(_DEBUG_)
				printf("Enter into an internal node %llu, count %d\n", node, node->count);
				int i;
				for(i=0; i<node->count*2 ; i++)
				{

if(_DEBUG_)
				  printf("i [%d] compare bounding box in an internal/leaf(level : %d) node index %llu, passed index %llu\n",i,node->level[i/2], node->index[i], passed_index);
					if( ( node->child[i] != NULL) &&
							( node->index[i] > passed_index) &&
							(RadixNode_SOA_Overlap(&query, node, i))
						)
					{
						break;
					}
				}

				if( i == node->count*2 )
				{
					passed_index = node->index[(node->count*2)-1];
if(_DEBUG_)
					printf("None of child nodes overlaps, set passed index to %llu, and move to root\n", passed_index);
					node = root;
				  break;
				}
				else
				{
if(_DEBUG_)
					printf("select %d's child node %llu \n",i, node->child[i]);
					node = node->child[ i ];
				}
			}

			bool one_more_chance = true;
			while( node->level[0] == AUXILIARY_LEAF_NODE)
			{
			  if( _DEBUG_ )
			  printf("Auxiliary leaf node, visits %llu ~ %llu node\n",node->index[0], node->index[node->count*2-1]);
if(_DEBUG_)
				printf("Leaf node, count %d \n",node->count);
				bool is_hit = false;
				for(int i=0;i<node->count*2; i+=2) // !!!
				{
if(_DEBUG_)
				  printf("i [%d] compare bounding box in leaf node, index is %llu\n",i, node->index[i]);
					if( RadixNode_SOA_Overlap(&query, node, i)) // !!!
					{
if(_DEBUG_)
					  printf("hit!!\n");
						hit++;
						is_hit = true;
					}
				}
				passed_index = node->index[(node->count*2)-1]; // !!!
if(_DEBUG_)
				printf("After scanning a leaf node, set the passed index to %llu\n", passed_index);

				if( passed_index == last_index )
				{
if(_DEBUG_)
				  printf("search operation is done\n");
					break;
				}
				else if( is_hit  || one_more_chance)
				{
if(_DEBUG_)
				    printf("get one more chance, keep scanning leaf nodes\n");
					 node++;
					 one_more_chance = false;
	
if(_DEBUG_)
				  printf("hit\n");
					node++;
				}
				else
				{
if(_DEBUG_)
				    printf("leaf -> root node\n");
				    node = root;
				}
			}

		}
if(_DEBUG_)
		printf("%dth search, hit is %d\n", s, hit);
	}
	printf("MPHR on CPU, Hit is %d\n", hit);
}
void TraverseHilbertRadixTreeUsingMPHR2(struct RadixTreeNode_SOA* root)
{
	//query
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
	}else
	{
		printf("Query Filename %s\n",queryFileName);
	}

	struct Rect query;

	int hit = 0;
	for( int s = 0 ; s < NUMSEARCH; s++ )
	{
		for(int d=0;d<NUMDIMS*2;d++)
			fread(&query.boundary[d], sizeof(float), 1, fp);

		//printf("Start Search Operation with query [%d]\n", s);

		unsigned long long passed_index = 0;
		unsigned long long last_index = root->index[(root->count*2)-1];
		//printf("Last index %llu\n", last_index);

		struct RadixTreeNode_SOA* node = root;

		while( passed_index < last_index )
		{	
			//printf("Passed index %llu  last index %llu\n", passed_index, last_index);

			int i;
			for(i=0; i<node->count*2; i++)
			{
				//printf("i [%d] compare bounding box in an internal/leaf(%d) node %llu, passed index %llu\n",i,node->level[i/2], node->index[i], passed_index);
				if( ( node->index[i] > passed_index) &&
						(RadixNode_SOA_Overlap(&query, node, i))
					)
				{
					break;
				}
			}

			if( i == node->count*2 )
			{
				passed_index = node->index[(node->count*2)-1];
				//printf("None of child nodes overlaps, set passed index to %llu\n", passed_index);
				node = root;
			}
			else
			{
				if( node->level[i/2] == LEAF_NODE)
				{
					hit++;
					passed_index = node->index[i];

					if( passed_index == last_index )
					{
						//printf("search operation is done\n");
						break;
					}
					else
					{
						//printf("leaf -> root node\n");
						node = root;
					}
				}
				else
				{
					//printf("select %d's child node\n",i);
					node = node->child[ i ];
				}
			}
		}
	}
	printf("Hit is %d\n", hit/2);
}
void settingRadixTrees(struct RadixTreeNode* n, const int reverse)
{
	while(!radix_q0.empty()){
		radix_q0.pop();
	}

	radix_q0.push(n);

	while(!radix_q0.empty()){
		struct RadixTreeNode* t = radix_q0.front();
		radix_q0.pop();

		if( t->level == INT_MAX)
			t->level = 0;
		else
			t->level = (reverse-(t->level));

		if( t->child[0] != NULL)
			radix_q0.push(t->child[0]);

		if( t->child[1] != NULL)
			radix_q0.push(t->child[1]);
	}
}


__global__ void globalRadixTreeLoadFromMem(char *buf, int partition_no, int NUMBLOCK, int PARTITIONED /* number of partitioned index*/, int numOfnodes)
{
	if( partition_no == 0 ){
		devNUMBLOCK = NUMBLOCK;
		deviceRadixRoot = (struct RadixTreeNode_SOA**) malloc( sizeof(struct RadixTreeNode_SOA*) * PARTITIONED );
	}

	deviceRadixRoot[partition_no] = (struct RadixTreeNode_SOA*) buf;
	devRadixTreeLoadFromMem(deviceRadixRoot[partition_no], buf, numOfnodes);
}
__device__ void devRadixTreeLoadFromMem(struct RadixTreeNode_SOA* root, char *buf, int numOfnodes)
{
	struct RadixTreeNode_SOA* node = root;

	for(int i=0; i<numOfnodes; i++)
	{
		for(int j=0; j<node->count*2; j++)
		{
				if( node->child[j] != NULL)
				{
					node->child[j] = (struct RadixTreeNode_SOA*) ((size_t) (node->child[j])+(size_t)buf);
				}
		}
		node++;
	}
}
__device__ void devicePrintRadixTree(struct RadixTreeNode_SOA* node)
{
		printf("Print Node %llu, count %d \n", node, node->count);
		for(int d=0; d<node->count ; d++)
		{
			printf("\nnode level %d", node->level[d]);
			printf("\nnode left index %llu", node->index[d*2]);
			printf("\nnode right index %llu", node->index[d*2+1]);
			printf("\nleft child node %llu", node->child[d*2]);
			printf("\nright child node %llu \n\n", node->child[d*2+1]);
		}
}
__global__ void globalPrintRadixTree(int p, int numOfNode, int a, int b, int *c)
{

	printf("root %llu\n", (struct RadixTreeNode_SOA*) deviceRadixRoot[p]);
	struct RadixTreeNode_SOA* node = (struct RadixTreeNode_SOA*) deviceRadixRoot[p];
	printf("node %llu\n", node);

	*c = a+b;
	printf("c is %d\n", *c);

	for(int i=0; i<numOfNode; i++)
	{
		devicePrintRadixTree(node);
		printf("\n\n");
		node++;
	}
}


