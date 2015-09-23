#include <mpes.h>
//#####################################################################
//####################### IMPLEMENTATIONS #############################
//#####################################################################


void MPES(int number_of_procs, int myrank)
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
	}else
	{
		printf("Query Filename %s\n",queryFileName);
	}

	long leafNode_offset[2];
	int numberOfLeafnodes[2];
	long* d_leafNode_offset;
	int* d_numberOfLeafnodes;

	if( BUILD_TYPE == 0)
	{
		leafNode_offset[0] = (int)(indexSize[0]/(long)PGSIZE) - number_of_node_in_level[0][0];
		leafNode_offset[1] = (int)(indexSize[1]/(long)PGSIZE) - number_of_node_in_level[1][0];

		numberOfLeafnodes[0] = number_of_node_in_level[0][0]; 
		numberOfLeafnodes[1] = number_of_node_in_level[1][0];

		cudaMalloc((void**)&d_leafNode_offset, sizeof(long)*2);
		cudaMemcpy( d_leafNode_offset, leafNode_offset, sizeof(long)*2, cudaMemcpyHostToDevice);

		cudaMalloc((void**)&d_numberOfLeafnodes, sizeof(int)*2);
		cudaMemcpy( d_numberOfLeafnodes, numberOfLeafnodes, sizeof(int)*2, cudaMemcpyHostToDevice);
	}


	int h_hit[NUMBLOCK];
	memset(h_hit, 0, sizeof(int)*NUMBLOCK);

	long t_hit = 0;
	int* d_hit;
	cudaMalloc( (void**) &d_hit,   NUMBLOCK*sizeof(int) );
	cudaMemcpy( d_hit, h_hit, sizeof(int)*NUMBLOCK, cudaMemcpyHostToDevice);


	if( PARTITIONED == 1 )
	{
		struct Rect* d_query;
#ifdef TP
		cudaMalloc( (void**) &d_query, NUMBLOCK*NUM_TP*sizeof(struct Rect) );
		struct Rect query[NUMBLOCK*NODECARD];
#else
		cudaMalloc( (void**) &d_query, NUMBLOCK*sizeof(struct Rect) );
		struct Rect query[NUMBLOCK];
#endif


		cudaEventRecord(start_event, 0);
		for( int i = 0 ; i < NUMSEARCH; ){
			int nBatch=0;
#ifdef TP
			for(nBatch=0; nBatch < NUMBLOCK*NUM_TP && i < NUMSEARCH; nBatch++, i++) {
#else
			for(nBatch=0; nBatch < NUMBLOCK && i < NUMSEARCH; nBatch++, i++) {
#endif
				for(int j=0;j<NUMDIMS*2;j++)
					fread(&query[nBatch].boundary[j], sizeof(float), 1, fp);
			}	
			cudaMemcpy(d_query, query, nBatch*sizeof(struct Rect), cudaMemcpyHostToDevice);

#ifdef ILP
			if( BUILD_TYPE == 0 )
				globalMPES_ILP<<<nBatch, NUMTHREADS>>>(d_query, d_hit, d_leafNode_offset, d_numberOfLeafnodes,  nBatch, boundary_of_trees, PARTITIONED);
			else
				globalMPES_ILP_BVH<<<nBatch, NUMTHREADS>>>(d_query, d_hit, nBatch, PARTITIONED);
#else
#ifdef TP
			if( BUILD_TYPE == 0 )
				globalMPES_TP<<<NUMBLOCK, NUM_TP>>>(d_query, d_hit, d_leafNode_offset, d_numberOfLeafnodes,  nBatch, boundary_of_trees, PARTITIONED);
			else
				globalMPES_BVH_TP<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, nBatch, PARTITIONED);
#else
			if( BUILD_TYPE == 0 )
				globalMPES<<<nBatch, NODECARD>>>(d_query, d_hit, d_leafNode_offset, d_numberOfLeafnodes,  nBatch, boundary_of_trees, PARTITIONED);
			else if( BUILD_TYPE==1 || BUILD_TYPE==2)
				globalMPES_BVH<<<nBatch, NODECARD>>>(d_query, d_hit, nBatch, PARTITIONED);
			else
				globalMPES_RadixArray<<<nBatch, NODECARD>>>(d_query, d_hit, nBatch, PARTITIONED);
#endif
#endif

#ifdef TP
			cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<NUMBLOCK;j++){
				t_hit += h_hit[j];
			}
#else
			cudaMemcpy(h_hit, d_hit, nBatch*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<nBatch;j++){
				t_hit += h_hit[j];
			}
#endif

		}

		cudaThreadSynchronize();
		cudaEventRecord(stop_event, 0) ;
		cudaEventSynchronize(stop_event) ;

		cudaFree( d_query);
	}else{

		struct Rect* d_query;
		cudaMalloc( (void**) &d_query, sizeof(struct Rect) );
		struct Rect query;


		cudaEventRecord(start_event, 0);
		for( int i = 0 ; i < NUMSEARCH; i++ ){
			for(int j=0;j<NUMDIMS*2;j++)
				fread(&query.boundary[j], sizeof(float), 1, fp);
			cudaMemcpy(d_query, &query, sizeof(struct Rect), cudaMemcpyHostToDevice);

#ifndef ILP
			if( BUILD_TYPE == 0 )
				globalMPES<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_leafNode_offset, d_numberOfLeafnodes, 1, boundary_of_trees, PARTITIONED);
			else
				globalMPES_BVH<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, 1, PARTITIONED);
#else
			if( BUILD_TYPE == 0 )
				globalMPES_ILP<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, d_leafNode_offset, d_numberOfLeafnodes,  1, boundary_of_trees, PARTITIONED);
			else
				globalMPES_ILP_BVH<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit,  1, PARTITIONED);
#endif
			cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<NUMBLOCK;j++){
				t_hit += h_hit[j];
			}
		}
		cudaThreadSynchronize();
		cudaEventRecord(stop_event, 0) ;
		cudaEventSynchronize(stop_event) ;

		cudaFree( d_query);

	}


	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	printf("MPES time          : %.3f ms\n", elapsed_time);
	printf("MPES HIT           : %lu\n",t_hit);
	printf("MPES HIT ratio(%)  : %5.5f\n",((t_hit/NUMSEARCH)/(float)(NUMDATA/100)));

	t_time[0] = elapsed_time;

	fclose(fp);

	cudaFree( d_leafNode_offset); 
	cudaFree( d_numberOfLeafnodes); 
	cudaFree( d_hit); 
}



__global__ void globalMPES(struct Rect * _query, int *b_hit, long * d_leafNode_offset, int* d_numberOfLeafnodes,  int mpiSEARCH, int boundary_index, int PARTITIONED)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;


	int tree_index;
	int partition_index=0;

	int block_init_position = bid;
	int block_incremental_value = devNUMBLOCK;

	if( PARTITIONED > 1 ){
		partition_index = bid;
		//task parallelism for partitioned index
		block_init_position = 0;
		block_incremental_value = 1;
	}
	if( boundary_index > bid )
		tree_index = 0;
	else
		tree_index = 1;


	long leafNode_offset = d_leafNode_offset[tree_index];
	int numberOfLeafnodes = d_numberOfLeafnodes[tree_index];

	__shared__ int t_hit[NODECARD];
	struct Rect query;
	b_hit[bid] = 0;
	t_hit[tid] = 0;

	__syncthreads();

	struct Node_SOA* first_leafNode = (struct Node_SOA*)deviceRoot[partition_index]+leafNode_offset;

	for( int n = block_init_position; n < mpiSEARCH ; n += block_incremental_value ) 
	{
		query = _query[n];
		struct Node_SOA* node = first_leafNode;

		for(int i=0; i<numberOfLeafnodes; i++ )
		{
			if( tid < node->count)
			{
				if( dev_Node_SOA_Overlap(&query, node, tid))
				{
					t_hit[tid]++;
				}
			}
			node++;
		}
	}

	__syncthreads();

	int N = NODECARD/2 + NODECARD%2;

	while(N > 1){
		if(tid<N){
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
			t_hit[tid+N] = 0;
		}
		N = N/2+N%2;
		__syncthreads();
	}

	if(tid == 0) {
		if(N==1){
			t_hit[0] = t_hit[0] + t_hit[1];
		}
		b_hit[bid] = t_hit[0];
	}
}

__global__ void globalMPES_ILP(struct Rect * _query, int *b_hit, long * d_leafNode_offset, int* d_numberOfLeafnodes,  int mpiSEARCH, int boundary_index, int PARTITIONED)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;


	int tree_index;
	int partition_index=0;

	int block_init_position = bid;
	int block_incremental_value = devNUMBLOCK;

	if( PARTITIONED > 1 ){
		partition_index = bid;
		//task parallelism for partitioned index
		block_init_position = 0;
		block_incremental_value = 1;
	}
	if( boundary_index > bid )
		tree_index = 0;
	else
		tree_index = 1;


	long leafNode_offset = d_leafNode_offset[tree_index];
	int numberOfLeafnodes = d_numberOfLeafnodes[tree_index];

	__shared__ int t_hit[NODECARD];
	__shared__ struct Rect query;
	b_hit[bid] = 0;


	for(int thread = tid; thread < NODECARD; thread+= NUMTHREADS)
		t_hit[thread] = 0;
	__syncthreads();

	struct Node_SOA* first_leafNode = (struct Node_SOA*)deviceRoot[partition_index]+leafNode_offset;

	for( int n = block_init_position; n < mpiSEARCH ; n += block_incremental_value ) 
	{
		query = _query[n];
		struct Node_SOA* node = first_leafNode;

		for(int i=0; i<numberOfLeafnodes; i++  )
		{
			for(int thread = tid; thread < node->count ; thread+= NUMTHREADS)
			{
				if( dev_Node_SOA_Overlap(&query, node, thread))
				{
					t_hit[thread]++;
				}
			}
			node++;
		}
	}
	__syncthreads();

	int N = NODECARD/2 + NODECARD%2;

	while(N > 1){
		for ( int thread = tid ; thread < N ; thread+= NUMTHREADS){
			t_hit[thread] = t_hit[thread] + t_hit[thread+N];
		}
		N = N/2+N%2;
		__syncthreads();
	}

	if(tid == 0) {
		if(N==1){
			t_hit[0] = t_hit[0] + t_hit[1];
		}
		b_hit[bid] = t_hit[0];
	}
}
__global__ void globalMPES_BVH(struct Rect * _query, int *b_hit, int mpiSEARCH, int PARTITIONED)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;


	int partition_index=0;
	int block_init_position = bid;
	int block_incremental_value = devNUMBLOCK;

	if( PARTITIONED > 1 ){
		partition_index = bid;
		//task parallelism for partitioned index
		block_init_position = 0;
		block_incremental_value = 1;
	}

	__shared__ int t_hit[NODECARD];
	__shared__ struct Rect query;
	b_hit[bid] = 0;
	t_hit[tid] = 0;

	__syncthreads();

	BVH_Node_SOA* leftmost_leafnode = (BVH_Node_SOA*) deviceBVHRoot[partition_index];

	while( leftmost_leafnode->level > 0 )
	{
		leftmost_leafnode = leftmost_leafnode->child[0];
	}


	for( int n = block_init_position; n < mpiSEARCH ; n += block_incremental_value ) 
	{
		query = _query[n];
		BVH_Node_SOA* node = leftmost_leafnode;

		while( node != 0x0 )
		{

			if( tid < node->count)
			{
				if( dev_BVH_Node_SOA_Overlap(&query, node , tid))
				{
					t_hit[tid]++;
				}
//				__syncthreads();
			}
			node = node->sibling;
		}
	}

	__syncthreads();

	int N = NODECARD/2 + NODECARD%2;

	while(N > 1){
		if(tid<N){
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
			t_hit[tid+N] = 0;
		}
		N = N/2+N%2;
		__syncthreads();
	}

	if(tid == 0) {
		if(N==1){
			t_hit[0] = t_hit[0] + t_hit[1];
		}
		b_hit[bid] = t_hit[0];
	}
}

__global__ void globalMPES_RadixArray(struct Rect * _query, int *b_hit, int mpiSEARCH,  int PARTITIONED)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int partition_index=0;

	int block_init_position = bid;
	int block_incremental_value = devNUMBLOCK;

	if( PARTITIONED > 1 ){
		partition_index = bid;
		//task parallelism for partitioned index
		block_init_position = 0;
		block_incremental_value = 1;
	}

	__shared__ int t_hit[NODECARD];
	struct Rect query;
	b_hit[bid] = 0;
	t_hit[tid] = 0;

	__syncthreads();

	// !!! TO DO : it can be eliminated by providing offset
	RadixTreeNode_SOA* leftmost_leafnode = (RadixTreeNode_SOA*) deviceRadixRoot[partition_index];
	while( leftmost_leafnode->level[0] > 0 )
	{
		leftmost_leafnode = leftmost_leafnode->child[0];
	}
	/*
	while( leftmost_leafnode->level[0] > 0 )
	{
		leftmost_leafnode++;
	}
	*/

	for( int n = block_init_position; n < mpiSEARCH ; n += block_incremental_value ) 
	{
		query = _query[n];
		RadixTreeNode_SOA* node = leftmost_leafnode;

		while( node->level[0] == 0 )
		{
			/*
			if( tid == 0 )
			{
				for(int i=0; i<NUMDIMS; i++)
				{
					printf("min q : %f\n",query.boundary[i]);
					printf("max q : %f\n",query.boundary[i+NUMDIMS]);
				}
			}
			*/

			/*
			if( tid == 0 )
			{
			  printf("Print NodeSOA %llu \n", node->index[0]);
				for(int d=0; d<node->count ; d++)
				{
					for(int r=0; r<NUMDIMS; r++)
					{
						printf("\nleft min boundary [%d] %f \n", r, node->boundary[(2*NODECARD*r)+d]);
						printf("\nleft max boundary [%d] %f \n", r, node->boundary[((NUMDIMS+r)*2*NODECARD)+d]);
					}
					printf("\nnode level %d", node->level[d]);
					printf("\nnode index %llu", node->index[d]);
					printf("\nnode count %llu", node->count);
					printf("\nleft child node %llu", node->child[d*2]);
					printf("\nright child node %llu \n\n", node->child[d*2+1]);
				}
			}
			__syncthreads();
			*/


			if( tid < node->count)
			{
				if( dev_RadixNode_SOA_Overlap(&query, node , tid*2))
				{
					t_hit[tid]++;
				}
				//				__syncthreads();
			}
			node++;
		}
	}

	__syncthreads();

	int N = NODECARD/2 + NODECARD%2;

	while(N > 1){
		if(tid<N){
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
			t_hit[tid+N] = 0;
		}
		N = N/2+N%2;
		__syncthreads();
	}

	if(tid == 0) {
		if(N==1){
			t_hit[0] = t_hit[0] + t_hit[1];
		}
		b_hit[bid] = t_hit[0];
	}
}

__global__ void globalMPES_ILP_BVH(struct Rect * _query, int *b_hit, int mpiSEARCH, int PARTITIONED )
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;


	int partition_index=0;

	int block_init_position = bid;
	int block_incremental_value = devNUMBLOCK;

	if( PARTITIONED > 1 ){
		partition_index = bid;
		//task parallelism for partitioned index
		block_init_position = 0;
		block_incremental_value = 1;
	}

	__shared__ int t_hit[NODECARD];
	__shared__ struct Rect query;
	b_hit[bid] = 0;


	for(int thread = tid; thread < NODECARD; thread+= NUMTHREADS)
		t_hit[thread] = 0;
	__syncthreads();

	BVH_Node_SOA* leftmost_leafnode = (BVH_Node_SOA*)deviceBVHRoot[partition_index];


	//should go down to leftmost leafnode
	while( leftmost_leafnode->level > 0 )
	{
		leftmost_leafnode = leftmost_leafnode->child[0];
	}


	for( int n = block_init_position; n < mpiSEARCH ; n += block_incremental_value ) 
	{
		query = _query[n];
		BVH_Node_SOA* node = leftmost_leafnode;

		while( node != 0x0 )
		{
			for(int thread = tid; thread < node->count ; thread+= NUMTHREADS)
			{
				if( dev_BVH_Node_SOA_Overlap(&query, node, thread ))
				{
					t_hit[thread]++;
				}
				__syncthreads();
			}
			node = node->sibling;
		}
	}
	__syncthreads();

	int N = NODECARD/2 + NODECARD%2;

	while(N > 1){
		for ( int thread = tid ; thread < N ; thread+= NUMTHREADS){
			t_hit[thread] = t_hit[thread] + t_hit[thread+N];
		}
		N = N/2+N%2;
		__syncthreads();
	}

	if(tid == 0) {
		if(N==1){
			t_hit[0] = t_hit[0] + t_hit[1];
		}
		b_hit[bid] = t_hit[0];
	}
}


__global__ void globalMPES_TP(struct Rect * _query, int *b_hit, long * d_leafNode_offset, int* d_numberOfLeafnodes,  int NSEARCH, int boundary_index, int PARTITIONED)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int query_id = (blockIdx.x*blockDim.x) + threadIdx.x;


	int tree_index=0;
	int partition_index=0;

	int block_init_position = query_id;
	int block_incremental_value = NSEARCH;

	/* I don't care about this at least TP case
	if( PARTITIONED > 1 ){
		partition_index = bid;
		//task parallelism for partitioned index
		block_init_position = 0;
		block_incremental_value = 1;
	}
	*/
//	if( boundary_index > bid )
//		tree_index = 0;
//	else
//		tree_index = 1;


	long leafNode_offset = d_leafNode_offset[tree_index];
	int numberOfLeafnodes = d_numberOfLeafnodes[tree_index];

	__shared__ int t_hit[NUM_TP];
	struct Rect query;
	b_hit[bid] = 0; 
	t_hit[tid] = 0; 

	__syncthreads();

	struct Node_SOA* first_leafNode = (struct Node_SOA*)deviceRoot[partition_index]+leafNode_offset;

	for( int n = block_init_position; n < NSEARCH ; n += block_incremental_value ) 
	{
		query = _query[n];
		struct Node_SOA* node = first_leafNode;

		for(int i=0; i<numberOfLeafnodes; i++ )
		{
			//if( tid < node->count)
			for(int t = 0; t < node->count; t++)
			{
				if( dev_Node_SOA_Overlap(&query, node, t))
				{
					t_hit[tid]++;
				}
			}
			node++;
		}
	}

	__syncthreads();

	int N = NUM_TP/2 + NUM_TP%2;
	while(N > 1){
		if ( tid < N )
		{
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
		}

		N = N/2 + N%2;
		__syncthreads();
	}

	if(tid == 0) {
		if(N==1){
			t_hit[0] = t_hit[0] + t_hit[1];
		}
		b_hit[bid] = t_hit[0];
	}
}
__global__ void globalMPES_BVH_TP(struct Rect * _query, int *b_hit, int NSEARCH, int PARTITIONED)
{
int bid = blockIdx.x;
	int tid = threadIdx.x;
	int query_id = (blockIdx.x*blockDim.x) + threadIdx.x;


	//int tree_index=0;
	int partition_index=0;

	int block_init_position = query_id;
	int block_incremental_value = NSEARCH;

	/* I don't care about this at least TP case
	if( PARTITIONED > 1 ){
		partition_index = bid;
		//task parallelism for partitioned index
		block_init_position = 0;
		block_incremental_value = 1;
	}
	*/
//	if( boundary_index > bid )
//		tree_index = 0;
//	else
//		tree_index = 1;


	__shared__ int t_hit[NUM_TP];
	struct Rect query;
	b_hit[bid] = 0; 
	t_hit[tid] = 0; 


	BVH_Node_SOA* first_leafNode = (BVH_Node_SOA*)deviceBVHRoot[partition_index];

	while( first_leafNode->level > 0 )
	{
		first_leafNode = first_leafNode->child[0];
	}

	__syncthreads();


	for( int n = block_init_position; n < NSEARCH ; n += block_incremental_value ) 
	{
		query = _query[n];
		BVH_Node_SOA* node = first_leafNode;

		while( node != 0x0)
		{
			//if( tid < node->count)
			for(int t = 0; t < node->count; t++)
			{
				if( dev_BVH_Node_SOA_Overlap(&query, node, t))
				{
					t_hit[tid]++;
				}
			}
			node++;
		}
	}

	__syncthreads();

	int N = NUM_TP/2 + NUM_TP%2;
	while(N > 1){
		if ( tid < N )
		{
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
		}

		N = N/2 + N%2;
		__syncthreads();
	}

	if(tid == 0) {
		if(N==1){
			t_hit[0] = t_hit[0] + t_hit[1];
		}
		b_hit[bid] = t_hit[0];
	}


}

