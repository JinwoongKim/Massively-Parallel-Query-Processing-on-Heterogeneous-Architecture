#include <mphr.h>

//#####################################################################
//####################### IMPLEMENTATIONS #############################
//#####################################################################


void MPHR(int number_of_procs, int myrank)
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
	} else
	{
		printf("Line %d : query file %s open \n",__LINE__, queryFileName);
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


	if( PARTITIONED == 1 )
	{
		struct Rect* d_query;
		cudaMalloc( (void**) &d_query, NUMBLOCK*sizeof(struct Rect) );
		struct Rect query[NUMBLOCK];

		cudaEventRecord(start_event, 0);
		for( int i = 0 ; i < NUMSEARCH;){
			int nBatch=0;
			for(nBatch=0; nBatch < NUMBLOCK && i < NUMSEARCH; nBatch++, i++) 
			{
				for(int j=0;j<2*NUMDIMS;j++){
					fread(&query[nBatch].boundary[j], sizeof(float), 1, fp);
				}	
			}
			cudaMemcpy(d_query, query, nBatch*sizeof(struct Rect), cudaMemcpyHostToDevice);

#ifndef ILP
			if( BUILD_TYPE == 0)
				globalMPHR<<<nBatch, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, nBatch, PARTITIONED );
			else if( BUILD_TYPE == 1 || BUILD_TYPE == 2)
				globalMPHR_BVH<<<nBatch, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, nBatch, PARTITIONED );
			else
			{
				globalMPHR_RadixTree<<<nBatch, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, nBatch, PARTITIONED );
			}
#else
			if( BUILD_TYPE == 0)
				globalMPHR_ILP<<<nBatch, NUMTHREADS>>>(d_query, d_hit, d_count, d_rootCount, nBatch, PARTITIONED );
			else
				globalMPHR_ILP_BVH<<<nBatch, NUMTHREADS>>>(d_query, d_hit, d_count, d_rootCount, nBatch, PARTITIONED );

#endif

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
	}
	else
	{
		struct Rect* d_query;
		cudaMalloc( (void**) &d_query, sizeof(struct Rect) );
		struct Rect query;


		cudaEventRecord(start_event, 0);
		for( int i = 0 ; i < NUMSEARCH; i++ ){
			//DEBUG
			for(int j=0;j<2*NUMDIMS;j++){
				fread(&query.boundary[j], sizeof(float), 1, fp);
			}	
			cudaMemcpy(d_query, &query, sizeof(struct Rect), cudaMemcpyHostToDevice);

#ifndef ILP
			if ( BUILD_TYPE == 0)
				globalMPHR<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, 1, PARTITIONED );
			else
				globalMPHR_BVH<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, 1, PARTITIONED );
#else
			if ( BUILD_TYPE == 0)
				globalMPHR_ILP<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, d_count, d_rootCount, 1, PARTITIONED );
			else
				globalMPHR_ILP_BVH<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, d_count, d_rootCount, 1, PARTITIONED );
#endif

			cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<NUMBLOCK;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
				t_rootCount += h_rootCount[j]; 
			}
		}
		cudaThreadSynchronize();
		cudaEventRecord(stop_event, 0) ;
		cudaEventSynchronize(stop_event) ;

		cudaFree( d_query );
	}
	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
	printf("MPHR time          : %.3f ms\n", elapsed_time);
	printf("MPHR HIT           : %lu\n",t_hit);
	printf("MPHR visited       : %.3f \n",(t_count)/(float)NUMSEARCH );
	printf("MPHR root visited  : %.3f \n", (t_rootCount)/(float)NUMSEARCH);
	printf("MPHR HIT ratio(%)  : %5.5f\n",((t_hit/NUMSEARCH)/(float)(NUMDATA/100)));

	t_time[2] = elapsed_time;
	t_visit[2] = (t_count)/(float)NUMSEARCH;
	t_rootVisit[2] = (t_rootCount)/(float)NUMSEARCH;


	fclose(fp);

	cudaFree( d_hit );
	cudaFree( d_count );
	cudaFree( d_rootCount );

}

__global__ void globalMPHR (struct Rect* _query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED  )
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
	__shared__ int childOverlap[NODECARD];
	__shared__ bool isHit;
	__shared__  struct Rect query;
	struct Node_SOA* node_ptr;

	t_hit[tid] = 0;
	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	struct Node_SOA* root = (struct Node_SOA*) deviceRoot[partition_index];

	int passed_Index;
	int last_Index;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_Index = 0;
		last_Index 	= root->index[root->count-1];

		node_ptr 			= root;
		if( tid == 0)
			rootCount[bid]++;
		__syncthreads();

		while( passed_Index < last_Index )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

				if( (tid < node_ptr->count) &&
						(node_ptr->index[tid] > passed_Index) &&
						(dev_Node_SOA_Overlap(&query, node_ptr, tid )))
				{
					childOverlap[tid] = tid;
				}
				else
				{
					childOverlap[tid] = NODECARD+1;
				}
				__syncthreads();


				// check if I am the leftmost
				// Gather the Overlap idex and compare

				int N = NODECARD/2 + NODECARD%2;
				while(N > 1){
					if ( tid < N ){
						if(childOverlap[tid] > childOverlap[tid+N] )  
							childOverlap[tid] = childOverlap[tid+N];
					}

					N = N/2+N%2;
					__syncthreads();
				}
				if( tid == 0) {
					if(N==1){
						if(childOverlap[0] > childOverlap[1])
							childOverlap[0] = childOverlap[1];
					}
				}
				__syncthreads();

				// none of the branches overlapped the query
				if( childOverlap[0] == ( NODECARD+1)) {
					passed_Index = node_ptr->index[node_ptr->count-1];

					node_ptr = root;
					if( tid == 0 ) rootCount[bid]++;

					break;
				}
				// there exists some overlapped node
				else{
					if( tid == 0 )
					{
						count[bid]++;
					}
					node_ptr = node_ptr->child[ childOverlap[0] ];
				}
				__syncthreads();
			}

			while( node_ptr->level == 0 )
			{

				isHit = false;

				if ( tid < node_ptr->count && dev_Node_SOA_Overlap(&query, node_ptr, tid))
				{
					t_hit[tid]++;
					isHit = true;
				}
				__syncthreads();

				passed_Index = node_ptr->index[node_ptr->count-1];

				//last leaf node

				if ( node_ptr->index[node_ptr->count-1] == last_Index )
					break;
				else if( isHit )
				{
					if( tid == 0 )
					{
						count[bid]++;
	
					}
					node_ptr++;
				}
				else
				{
					if( tid == 0 )
					{
						rootCount[bid]++;
					}
					node_ptr = root;
				}
				__syncthreads();
			}
		}
	}
	__syncthreads();
	int N = NODECARD/2 + NODECARD%2;
	while(N > 1){
		if ( tid < N ){
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
		}

		N = N/2 + N%2;
		__syncthreads();
	}

	if(tid==0) {
		if(N==1) 
			hit[bid] = t_hit[0] + t_hit[1];
		else
			hit[bid] = t_hit[0];
	}

	__syncthreads();

}
__global__ void globalMPHR_ILP(struct Rect* _query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED)
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
	__shared__ int childOverlap[NODECARD];
	__shared__ bool isHit;
	__shared__  struct Rect query;
	struct Node_SOA* node_ptr;

	for ( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS)
		t_hit[thread] = 0;

	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	struct Node_SOA* root = (struct Node_SOA*) deviceRoot[partition_index];

	int passed_hIndex;
	int last_hIndex;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_hIndex = 0;
		last_hIndex 	= root->index[root->count-1];

		node_ptr 			= root;
		if( tid == 0)
		{
			rootCount[bid]++;
		}
		__syncthreads();

		while( passed_hIndex < last_hIndex )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

//			if( tid == 0 )
//				printf("Internal node %d\n",node_ptr->level);


				for ( int thread = tid ; thread < NODECARD ; thread+=NUMTHREADS)
				{
					if( (thread < node_ptr->count) &&
							(node_ptr->index[thread]> passed_hIndex) &&
							(dev_Node_SOA_Overlap(&query, node_ptr, thread))){
						childOverlap[thread] = thread;
					}
					else
					{
						childOverlap[thread] = NODECARD+1;
					}
				}
				__syncthreads();


				// check if I am the leftmost
				// Gather the Overlap idex and compare

				int N = NODECARD/2 + NODECARD%2;
				while(N > 1){
					for ( int thread = tid ; thread < N ; thread+=NUMTHREADS)
					{
						if(childOverlap[thread] > childOverlap[thread+N] )  
							childOverlap[thread] = childOverlap[thread+N];
					}
					N = N/2+N%2;
					__syncthreads();
				}
				if( tid == 0) {
					if(N==1){
						if(childOverlap[0] > childOverlap[1])
							childOverlap[0] = childOverlap[1];
					}
				}
				__syncthreads();

				// none of the branches overlapped the query
				if( childOverlap[0] == ( NODECARD+1)) {
					passed_hIndex = node_ptr->index[node_ptr->count-1];

					if( tid == 0 )
					{
						rootCount[bid]++;
					}
					node_ptr = root;

					break;
				}
				// there exists some overlapped node
				else{
					if( tid == 0 )
					{
						count[bid]++;
					}
					node_ptr = node_ptr->child[ childOverlap[0] ];
				}
				__syncthreads();
			}



			while( node_ptr->level == 0 )
			{

			//	if( tid == 0 )
			//		printf("Leaf node %d\n",node_ptr->level);

				isHit = false;


				for ( int thread = tid ; thread < NODECARD ; thread+=NUMTHREADS){
					if ( (thread < node_ptr->count) &&
							dev_Node_SOA_Overlap(&query, node_ptr, thread)
						 )
					{
						t_hit[thread]++;
						isHit = true;
					}
				}
				__syncthreads();

				passed_hIndex = node_ptr->index[node_ptr->count-1];

				//last leaf node

				if ( node_ptr->index[node_ptr->count-1] == last_hIndex )
					break;
				else if( isHit )
				{
					if( tid == 0 )
					{
						count[bid]++;
					}
					node_ptr++;
				}
				else
				{
					if( tid == 0 )
					{
						rootCount[bid]++;
					}
					node_ptr = root;
				}
				__syncthreads();
			}
		}
	}
	__syncthreads();
	int N = NODECARD/2 + NODECARD%2;
	while(N > 1){
		for ( int thread = tid ; thread < N ; thread+=NUMTHREADS){
			t_hit[thread] = t_hit[thread] + t_hit[thread+N];
		}

		N = N/2 + N%2;
		__syncthreads();
	}

	if(tid==0) {
		if(N==1) 
			hit[bid] = t_hit[0] + t_hit[1];
		else
			hit[bid] = t_hit[0];
	}

	__syncthreads();

}
__global__ void globalMPHR_BVH (struct Rect* _query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED)
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
	__shared__ int childOverlap[NODECARD];
	__shared__ bool isHit;
	__shared__  struct Rect query;
	BVH_Node_SOA* node_ptr;

	t_hit[tid] = 0;
	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	BVH_Node_SOA* root =  (BVH_Node_SOA*) deviceBVHRoot[partition_index];

	unsigned long long passed_mortonCode;
	unsigned long long last_mortonCode;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_mortonCode = 0;
		last_mortonCode 	= root->index[root->count-1];

		if( tid == 0 )
		{
			rootCount[bid]++;
		}
		node_ptr 			= root;
		__syncthreads();

		while( passed_mortonCode < last_mortonCode )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

				if( (tid < node_ptr->count) &&
						(node_ptr->index[tid] > passed_mortonCode) &&
						(dev_BVH_Node_SOA_Overlap(&query, node_ptr, tid)))
				{
					childOverlap[tid] = tid;
				}
				else
				{
					childOverlap[tid] = NODECARD+1;
				}
				__syncthreads();


				// check if I am the leftmost
				// Gather the Overlap idex and compare

				int N = NODECARD/2 + NODECARD%2;
				while(N > 1){
					if ( tid < N ){
						if(childOverlap[tid] > childOverlap[tid+N] )  
							childOverlap[tid] = childOverlap[tid+N];
					}

					N = N/2+N%2;
					__syncthreads();
				}
				if( tid == 0) {
					if(N==1){
						if(childOverlap[0] > childOverlap[1])
							childOverlap[0] = childOverlap[1];
					}
				}
				__syncthreads();

				// none of the branches overlapped the query
				if( childOverlap[0] == ( NODECARD+1)) {
					passed_mortonCode = node_ptr->index[node_ptr->count-1];

					if( tid == 0 )
					{
						rootCount[bid]++;
					}
					node_ptr = root;
					break;
				}
				// there exists some overlapped node
				else{
					if( tid == 0 )
					{
						count[bid]++;
					}
					node_ptr = node_ptr->child[ childOverlap[0] ];
				}
				__syncthreads();
			}


			while( node_ptr->level == 0 )
			{
				isHit = false;
				__syncthreads();

				if ( tid < node_ptr->count && dev_BVH_Node_SOA_Overlap(&query, node_ptr, tid))
				{
					t_hit[tid]++;
					isHit = true;
				}
				__syncthreads();

				passed_mortonCode = node_ptr->index[node_ptr->count-1];

				//last leaf node

				if ( node_ptr->index[node_ptr->count-1]== last_mortonCode )
					break;
				else if( isHit )
				{
					if( tid == 0 )
					{
						count[bid]++;
					}
					node_ptr = node_ptr->sibling;
				}
				else
				{
					if( tid == 0 )
					{
						rootCount[bid]++;
					}
					node_ptr = root;
				}

				__syncthreads();
			}
		}
	}
	__syncthreads();
	int N = NODECARD/2 + NODECARD%2;
	while(N > 1){
		if ( tid < N ){
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
		}

		N = N/2 + N%2;
		__syncthreads();
	}

	if(tid==0) {
		if(N==1) 
			hit[bid] = t_hit[0] + t_hit[1];
		else
			hit[bid] = t_hit[0];
	}

	__syncthreads();

}
__global__ void globalMPHR_ILP_BVH(struct Rect* _query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED)
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
	__shared__ int childOverlap[NODECARD];
	__shared__ bool isHit;
	__shared__  struct Rect query;
	BVH_Node_SOA* node_ptr;

	for ( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS)
		t_hit[thread] = 0;

	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	BVH_Node_SOA* root = (BVH_Node_SOA*)deviceBVHRoot[partition_index];

	unsigned long long passed_mortonCode;
	unsigned long long last_mortonCode;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_mortonCode = 0;
		last_mortonCode 	= root->index[root->count-1];

		if( tid == 0 )
		{
			rootCount[bid]++;
		}
		node_ptr 			= root;

		__syncthreads();
		while( passed_mortonCode < last_mortonCode )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

				for ( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS){
					if( (thread < node_ptr->count ) &&
							(node_ptr->index[thread]> passed_mortonCode) &&
							(dev_BVH_Node_SOA_Overlap(&query, node_ptr, thread))){
						childOverlap[thread] = thread;
					}
					else
					{
						childOverlap[thread] = NODECARD+1;
					}
				}
				__syncthreads();


				// check if I am the leftmost
				// Gather the Overlap idex and compare

				int N = NODECARD/2 + NODECARD%2;
				while(N > 1){
					for ( int thread = tid ; thread < N ; thread+=NUMTHREADS){
						if(childOverlap[thread] > childOverlap[thread+N] )  
							childOverlap[thread] = childOverlap[thread+N];
					}
					N = N/2+N%2;
					__syncthreads();
				}
				if( tid == 0) {
					if(N==1){
						if(childOverlap[0] > childOverlap[1])
							childOverlap[0] = childOverlap[1];
					}
				}
				__syncthreads();

				// none of the branches overlapped the query
				if( childOverlap[0] == ( NODECARD+1)) {
					passed_mortonCode = node_ptr->index[node_ptr->count-1];

					if( tid == 0 )
					{
						rootCount[bid]++;
					}
					node_ptr = root;
					break;
				}
				// there exists some overlapped node
				else{
					if( tid == 0 )
					{
						count[bid]++;
					}
					node_ptr = node_ptr->child[ childOverlap[0] ];
				}
				__syncthreads();
			}



			while( node_ptr->level == 0 )
			{
				isHit = false;


				for ( int thread = tid ; thread < node_ptr->count ; thread+=NUMTHREADS){
					if ( dev_BVH_Node_SOA_Overlap(&query, node_ptr, thread))
					{
						t_hit[thread]++;
						isHit = true;
					}
				}
				__syncthreads();

				passed_mortonCode = node_ptr->index[node_ptr->count-1];

				//last leaf node

				if ( node_ptr->index[node_ptr->count-1] == last_mortonCode )
					break;
				else if( isHit )
				{
					if( tid == 0 )
					{
						count[bid]++;
					}
					node_ptr = node_ptr->sibling;
				}
				else
				{
					if( tid == 0 )
					{
						rootCount[bid]++;
					}
					node_ptr = root;
				}

				__syncthreads();
			}
		}
	}
	__syncthreads();
	int N = NODECARD/2 + NODECARD%2;
	while(N > 1){
		for ( int thread = tid ; thread < N ; thread+=NUMTHREADS){
			t_hit[thread] = t_hit[thread] + t_hit[thread+N];
		}

		N = N/2 + N%2;
		__syncthreads();
	}

	if(tid==0) {
		if(N==1) 
			hit[bid] = t_hit[0] + t_hit[1];
		else
			hit[bid] = t_hit[0];
	}

	__syncthreads();

}
__global__ void globalMPHR_RadixTree (struct Rect* _query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED)
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
	__shared__ int childOverlap[NODECARD*2];
	__shared__ bool isHit;
	__shared__  struct Rect query;

	t_hit[tid] = 0;
	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	RadixTreeNode_SOA* root =  (RadixTreeNode_SOA*) deviceRadixRoot[partition_index];
	RadixTreeNode_SOA* node = root;

	unsigned long long passed_index;
	unsigned long long last_index;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_index = 0;
		last_index 	= root->index[(root->count*2)-1];

		if( tid == 0 )
		{
			rootCount[bid]++;
		}
		node = root;
		__syncthreads();

		while( passed_index < last_index )
		{	
			//find out left most child node till leaf level
			while( node->level[0] > 0 ) {

				int thread = tid;
				if( (thread<node->count ) &&
						(node->child[thread] != NULL) &&
						(node->index[thread] > passed_index) &&
						(dev_RadixNode_SOA_Overlap(&query, node, thread)))
				{
					childOverlap[thread] = thread;
				}
				else
				{
					childOverlap[thread] = 2*NODECARD;
				}
				thread+=NODECARD;

				if( (thread<node->count*2 ) &&
						(node->child[thread] != NULL) &&
						(node->index[thread] > passed_index) &&
						(dev_RadixNode_SOA_Overlap(&query, node, thread)))
				{
					childOverlap[thread] = thread;
				}
				else
				{
					childOverlap[thread] = 2*NODECARD;
				}

				/*
				for(int thread=tid; thread < NODECARD*2; thread+=NODECARD)
				{
					if( (thread<node->count*2 ) &&
					    (node->child[thread] != NULL) &&
							(node->index[thread] > passed_index) &&
							(dev_RadixNode_SOA_Overlap(&query, node, thread)))
					{
						childOverlap[thread] = thread;
					}
					else
					{
						childOverlap[thread] = 2*NODECARD;
					}
				}
				*/
				__syncthreads();

				// check if I am the leftmost
				// Gather the Overlap idex and compare

				int N = NODECARD; 
				while(N > 1){
					if ( tid < N ){
						if(childOverlap[tid] > childOverlap[tid+N] )  
							childOverlap[tid] = childOverlap[tid+N];
					}

					N = N/2;
					__syncthreads();
				}
				if( tid == 0) {
					if(N==1){
						if(childOverlap[0] > childOverlap[1])
							childOverlap[0] = childOverlap[1];
					}
				}
				__syncthreads();

				// none of the branches overlapped the query
				if( childOverlap[0] == (2*NODECARD) ) 
				{
					passed_index = node->index[(node->count*2)-1];

					if( tid == 0 )
					{
						rootCount[bid]++;
					}
					node = root;
					//node = leftmost_leafnode;
					break;
				}
				// there exists some overlapped node
				else{
					if( tid == 0 )
					{
						count[bid]++;
					}
					node = node->child[ childOverlap[0] ];
				}
				__syncthreads();
			}

			while( node->level[0] == 0)
			{
				isHit = false;
				__syncthreads();

				//if ( tid < node->count && dev_RadixNode_SOA_Overlap(&query, node, tid*2))
				if ( tid < node->count && dev_RadixNode_SOA_Overlap(&query, node, tid))
				{
					t_hit[tid]++;
					isHit = true;
				}
				__syncthreads();

				passed_index = node->index[(node->count*2)-1];

				//last leaf node

				if( tid == 0 )
				{
					count[bid]++;
				}

				if ( passed_index == last_index )
					break;
 				else if( isHit )
 				{
 					if( tid == 0 )
 					{
 						count[bid]++;
 					}
 					node++;
 				}
				else
				{
					if( tid == 0 )
					{
						rootCount[bid]++;
					}
					//leftmost_leafnode = node;
					node = root;
				}

				__syncthreads();
			}

		} // end of while

	}// end of for
	__syncthreads();
	int N = NODECARD/2 + NODECARD%2;
	while(N > 1){
		if ( tid < N ){
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
		}

		N = N/2 + N%2;
		__syncthreads();
	}

	if(tid==0) {
		if(N==1) 
			hit[bid] = t_hit[0] + t_hit[1];
		else
			hit[bid] = t_hit[0];
	}

	__syncthreads();

}

void MPHR2(int number_of_procs, int myrank)
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
	memset(h_hit, 0, sizeof(int)*NUMBLOCK);
	int h_count[NUMBLOCK];
	int h_rootCount[NUMBLOCK];

	long t_hit = 0;
	long t_count = 0;
	int t_rootCount = 0;

	int* d_hit;
	int* d_count;
	int* d_rootCount;

	cudaMalloc( (void**) &d_hit,   NUMBLOCK*sizeof(int) );
	cudaMemcpy(d_hit, h_hit, sizeof(int)*NUMBLOCK, cudaMemcpyHostToDevice);
	cudaMalloc( (void**) &d_count, NUMBLOCK*sizeof(int) );
	cudaMalloc( (void**) &d_rootCount, NUMBLOCK*sizeof(int) );

	long leafNode_offset[2];
	long extendLeafNode_offset[2];
	long* d_leafNode_offset;
	long* d_extendLeafNode_offset;

	if( BUILD_TYPE == 0)
	{
		leafNode_offset[0] = (indexSize[0]/PGSIZE) - number_of_node_in_level[0][0];
		leafNode_offset[1] = (indexSize[1]/PGSIZE) - number_of_node_in_level[1][0];
		extendLeafNode_offset[0] = leafNode_offset[0] - number_of_node_in_level[0][1];
		extendLeafNode_offset[1] = leafNode_offset[1] - number_of_node_in_level[1][1];

		cudaMalloc((void**)&d_leafNode_offset, sizeof(long)*2);
		cudaMalloc((void**)&d_extendLeafNode_offset, sizeof(long)*2);

		cudaMemcpy( d_leafNode_offset, leafNode_offset, sizeof(long)*2, cudaMemcpyHostToDevice);
		cudaMemcpy( d_extendLeafNode_offset, extendLeafNode_offset, sizeof(long)*2, cudaMemcpyHostToDevice);

	}

	if( PARTITIONED == 1 )
	{
		struct Rect* d_query;
#ifdef TP
		cudaMalloc( (void**) &d_query, NUMBLOCK*NUM_TP*sizeof(struct Rect) );
		struct Rect query[NUMBLOCK*NUM_TP];
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
				//DEBUG
				for(int j=0;j<2*NUMDIMS;j++){
					fread(&query[nBatch].boundary[j], sizeof(float), 1, fp);
				}	
			}
			cudaMemcpy(d_query, query, nBatch*sizeof(struct Rect), cudaMemcpyHostToDevice);

#ifdef ILP
			if( BUILD_TYPE == 0)
				globalMPHR2_ILP<<<nBatch,NUMTHREADS >>>(d_query, d_hit, d_count, d_leafNode_offset, d_extendLeafNode_offset , d_rootCount, nBatch, boundary_of_trees, PARTITIONED );
			else
				globalMPHR2_ILP_BVH<<<nBatch, NUMTHREADS>>>(d_query, d_hit, d_count, d_rootCount, nBatch, PARTITIONED );
#else
#ifdef TP
			if( BUILD_TYPE == 0)
				globalMPHR2_TP<<<NUMBLOCK, NUM_TP>>>(d_query, d_hit, d_count, d_leafNode_offset, d_extendLeafNode_offset , d_rootCount, nBatch, boundary_of_trees, PARTITIONED );
			else
				globalMPHR2_BVH_TP<<<NUMBLOCK, NUM_TP>>>(d_query, d_hit, d_count, d_rootCount, nBatch, PARTITIONED );
#else
			if( BUILD_TYPE == 0)
				globalMPHR2<<<nBatch, NODECARD>>>(d_query, d_hit, d_count, d_leafNode_offset, d_extendLeafNode_offset , d_rootCount, nBatch, boundary_of_trees, PARTITIONED );
			else
				globalMPHR2_BVH<<<nBatch, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, nBatch, PARTITIONED );
#endif
#endif

#ifdef TP
			cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<NUMBLOCK;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
				t_rootCount += h_rootCount[j]; 
			}

#else
			cudaMemcpy(h_hit, d_hit, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, nBatch*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<nBatch;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
				t_rootCount += h_rootCount[j]; 
			}


#endif
		}

		cudaThreadSynchronize();
		cudaEventRecord(stop_event, 0) ;
		cudaEventSynchronize(stop_event) ;
		cudaFree( d_query );
	}
	else
	{
		struct Rect* d_query;
		cudaMalloc( (void**) &d_query, sizeof(struct Rect) );
		struct Rect query;


		cudaEventRecord(start_event, 0);
		for( int i = 0 ; i < NUMSEARCH; i++ ){
			//DEBUG
			for(int j=0;j<2*NUMDIMS;j++){
				fread(&query.boundary[j], sizeof(float), 1, fp);
			}	
			cudaMemcpy(d_query, &query, sizeof(struct Rect), cudaMemcpyHostToDevice);

#ifndef ILP
			if( BUILD_TYPE == 0)
				globalMPHR2<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_leafNode_offset, d_extendLeafNode_offset , d_rootCount, 1, boundary_of_trees, PARTITIONED );
			else
				globalMPHR2_BVH<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, 1, PARTITIONED );
#else
			if( BUILD_TYPE == 0)
				globalMPHR2_ILP<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, d_count, d_leafNode_offset, d_extendLeafNode_offset , d_rootCount, 1, boundary_of_trees, PARTITIONED );
			else
				globalMPHR2_ILP_BVH<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, d_count, d_rootCount, 1, PARTITIONED );
#endif

			cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<NUMBLOCK;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
				t_rootCount += h_rootCount[j]; 
//				if( h_count[j] > 2 || h_rootCount[j] > 0 )
//				{
//					printf("%d Internal Count %d\n", j, h_count[j]);
//					printf("%d LeafNode Count %d\n", j, h_rootCount[j]);
//				}
			}
		}

		cudaThreadSynchronize();
		cudaEventRecord(stop_event, 0) ;
		cudaEventSynchronize(stop_event) ;
		cudaFree( d_query );
	}

	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	printf("MPHR2 time          : %.3f ms\n", elapsed_time);
	printf("MPHR2 HIT           : %lu\n",t_hit);
	printf("MPHR2 visited       : %.3f \n",(t_count)/(float)NUMSEARCH );
	printf("MPHR2 root visited  : %.3f \n", (t_rootCount)/(float)NUMSEARCH);
	printf("MPHR2 HIT ratio(%)  : %5.5f\n",((t_hit/NUMSEARCH)/(float)(NUMDATA/100)));

	t_time[3] = elapsed_time;

	t_visit[3] = (t_count)/(float)NUMSEARCH;
	t_rootVisit[3] = (t_rootCount)/(float)NUMSEARCH;


	fclose(fp);

	cudaFree( d_hit );
	cudaFree( d_count );
	cudaFree( d_rootCount );


}


__global__ void globalMPHR2(struct Rect* _query, int * hit, int* count , long* _leafNode_offset, long* _extendLeafNode_offset , int* rootCount, int mpiSEARCH, int boundary_index, int PARTITIONED)
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


	long leafNode_offset = _leafNode_offset[tree_index];
	long extendLeafNode_offset = _extendLeafNode_offset[tree_index];

	__shared__ int t_hit[NODECARD]; 
	__shared__ int childOverlap[NODECARD];
	__shared__ bool isHit;
	__shared__  struct Rect query;
	struct Node_SOA* node_ptr;


	t_hit[tid] = 0;
	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	struct Node_SOA* root = (struct Node_SOA*) deviceRoot[partition_index];
	struct Node_SOA* leafNode_ptr = (struct Node_SOA*) ( (char*) root+(PGSIZE*leafNode_offset) );
	struct Node_SOA* extendNode_ptr = (struct Node_SOA*) ( (char*) root+(PGSIZE*extendLeafNode_offset) );

	__syncthreads();


	int passed_hIndex;
	int last_hIndex;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_hIndex = 0;
		last_hIndex 	= root->index[root->count-1];

		node_ptr 			= root;
		if( tid == 0 )
		{
			rootCount[bid]++;
		}
		__syncthreads();

		while( passed_hIndex < last_hIndex )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

				if( (tid < node_ptr->count) &&
						(node_ptr->index[tid]> passed_hIndex) &&
						(dev_Node_SOA_Overlap(&query, node_ptr, tid)))
				{
					childOverlap[tid] = tid;
				}
				else
				{
					childOverlap[tid] = NODECARD+1;
				}
				__syncthreads();


				// check if I am the leftmost
				// Gather the Overlap idex and compare

				int N = NODECARD/2 + NODECARD%2;
				while(N > 1){
					if ( tid < N ){
						if(childOverlap[tid] > childOverlap[tid+N] )  
							childOverlap[tid] = childOverlap[tid+N];
					}
					N = N/2+N%2;
					__syncthreads();
				}
				if( tid == 0) {
					if(N==1){
						if(childOverlap[0] > childOverlap[1])
							childOverlap[0] = childOverlap[1];
					}
				}
				__syncthreads();

				// none of the branches overlapped the query
				if( childOverlap[0] == ( NODECARD+1)) {
					passed_hIndex = node_ptr->index[node_ptr->count-1];

					node_ptr = root;
					if( tid == 0 )
					{
						rootCount[bid]++;
					}
					break;
				}
				// there exists some overlapped node
				else{
					node_ptr = node_ptr->child[ childOverlap[0] ];
					if( tid == 0 )
					{
							count[bid]++;
					}
				}
				__syncthreads();
			}


			while( node_ptr->level == 0 )
			{

				isHit = false;

				if ( tid < node_ptr->count && dev_Node_SOA_Overlap(&query, node_ptr, tid))
				{
					t_hit[tid]++;
					isHit = true;
				}
				__syncthreads();

				passed_hIndex = node_ptr->index[node_ptr->count-1];

				//last leaf node

				if ( node_ptr->index[node_ptr->count-1] == last_hIndex )
					break;
				else if( isHit )
				{
					node_ptr++;
					if( tid == 0 )
					{
						count[bid]++;
					}
					__syncthreads();
				}
				else
				{
					node_ptr = extendNode_ptr + ( ( node_ptr - leafNode_ptr) / NODECARD) ;
					if( tid == 0 )
					{
						if( node_ptr == root)
							rootCount[bid]++;
						else
							count[bid]++;
					}
					__syncthreads();
				}
			}
		}
	}
	__syncthreads();
	int N = NODECARD/2 + NODECARD%2;
	while(N > 1){
		if ( tid < N )
		{
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
		}

		N = N/2 + N%2;
		__syncthreads();
	}

	if(tid==0) {
		if(N==1) 
			hit[bid] = t_hit[0] + t_hit[1];
		else
			hit[bid] = t_hit[0];
	}

}


__global__ void globalMPHR2_ILP(struct Rect* _query, int * hit, int* count , long* _leafNode_offset, long* _extendLeafNode_offset , int* rootCount, int mpiSEARCH, int boundary_index, int PARTITIONED)
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


	long leafNode_offset = _leafNode_offset[tree_index];
	long extendLeafNode_offset = _extendLeafNode_offset[tree_index];

	__shared__ int t_hit[NODECARD]; 
	__shared__ int childOverlap[NUMTHREADS];
	__shared__ bool isHit;
	__shared__  struct Rect query;

	for ( int thread = tid ; thread < NODECARD ; thread+=NUMTHREADS)
		t_hit[thread] = 0;

	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	struct Node_SOA* root = (struct Node_SOA*)deviceRoot[partition_index];
	struct Node_SOA* leafNode_ptr   = (struct Node_SOA*)((char*) root+(PGSIZE*leafNode_offset ));
	struct Node_SOA* extendNode_ptr = (struct Node_SOA*) ( (char*) root+(PGSIZE*extendLeafNode_offset) );

	int passed_hIndex;
	int last_hIndex;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];

		passed_hIndex = 0;
		last_hIndex 	= root->index[root->count-1];

		struct Node_SOA* node_ptr 			= root;
		if( tid == 0 ) rootCount[bid]++;
		__syncthreads();

		while( passed_hIndex < last_hIndex )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

				__shared__ bool isOverlap;
				isOverlap = false;
//				__syncthreads();
				for ( int thread = tid ; thread < NODECARD ; thread+=NUMTHREADS){
					if(
							(thread < node_ptr->count) &&
							(node_ptr->index[thread]> passed_hIndex) &&
							(dev_Node_SOA_Overlap(&query, node_ptr, thread)))
					{
						childOverlap[tid] = thread;
						isOverlap = true;
					}
					else
					{
						childOverlap[tid] = NODECARD+1;
					}
					__syncthreads();
					if( isOverlap )
						break;
				}
				__syncthreads();

				int N = NUMTHREADS/2 + NUMTHREADS%2;
				while(N > 1){
					if( tid < N ){
						if(childOverlap[tid] > childOverlap[tid+N] )  
							childOverlap[tid] = childOverlap[tid+N];
					}
					N = N/2+N%2;
					__syncthreads();
				}

				if( tid == 0) {
					if(N==1){
						if(childOverlap[0] > childOverlap[1]) {
							childOverlap[0] = childOverlap[1];
						}
					}
				}
				__syncthreads();



				// none of the branches overlapped the query
				if( childOverlap[0] == ( NODECARD+1)) {
					passed_hIndex = node_ptr->index[node_ptr->count-1];

					if( tid == 0 ) rootCount[bid]++;
					node_ptr = root;
					__syncthreads();
					break;
				}
				// there exists some overlapped node
				else{
					node_ptr = node_ptr->child[ childOverlap[0] ];
					if( tid == 0 ) count[bid]++;
				}
				__syncthreads();
			}


			while( node_ptr->level == 0 )
			{
				isHit = false;
				__syncthreads();

				for ( int thread = tid ; thread < NODECARD; thread+= NUMTHREADS )
				{
					if ( thread < node_ptr->count && dev_Node_SOA_Overlap(&query, node_ptr, thread))
					{
						t_hit[thread]++;
						isHit = true;
					}
				}
				__syncthreads();

				passed_hIndex = node_ptr->index[node_ptr->count-1];

				//last leaf node

				if ( node_ptr->index[node_ptr->count-1]== last_hIndex )
					break;

				else if( isHit )
				{
					if( tid == 0 ) count[bid]++;
					node_ptr++;
				}
				else
				{
					node_ptr = extendNode_ptr + ( ( node_ptr - leafNode_ptr) / NODECARD) ;
					if( tid == 0 )
					{
						if( node_ptr == root)
							rootCount[bid]++;
						else
							count[bid]++;
					}
				}

				__syncthreads();
			}
		}
	}
	__syncthreads();

	int N = NODECARD/2 + NODECARD%2;
	while(N > 1){
		for ( int thread = tid ; thread < N ; thread+=NUMTHREADS){
			t_hit[thread] = t_hit[thread] + t_hit[thread+N];
		}

		N = N/2 + N%2;
		__syncthreads();
	}

	if(tid==0) {
		if(N==1) 
			hit[bid] = t_hit[0] + t_hit[1];
		else
			hit[bid] = t_hit[0];
	}

	__syncthreads();


}

__global__ void globalMPHR2_BVH(struct Rect* _query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED)
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
	__shared__ int childOverlap[NODECARD];
	__shared__ bool isHit;
	__shared__  struct Rect query;

	t_hit[tid] = 0;
	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	BVH_Node_SOA* root = (BVH_Node_SOA*)deviceBVHRoot[partition_index];

	int passed_mortonCode;
	int last_mortonCode;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_mortonCode = 0;
		last_mortonCode 	= root->index[root->count-1];

		BVH_Node_SOA* node_ptr 			= root;
		if( tid == 0 ) rootCount[bid]++;
		__syncthreads();

		while( passed_mortonCode < last_mortonCode )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

				if( (tid < node_ptr->count) &&
						(node_ptr->index[tid]> passed_mortonCode) &&
						(dev_BVH_Node_SOA_Overlap(&query, node_ptr, tid)))
				{
					childOverlap[tid] = tid;
				}
				else
				{
					childOverlap[tid] = NODECARD+1;
				}
				__syncthreads();


				// check if I am the leftmost
				// Gather the Overlap idex and compare

				int N = NODECARD/2 + NODECARD%2;
				while(N > 1){
					if ( tid < N ){
						if(childOverlap[tid] > childOverlap[tid+N] )  
							childOverlap[tid] = childOverlap[tid+N];
					}
					N = N/2+N%2;
					__syncthreads();
				}
				if( tid == 0) {
					if(N==1){
						if(childOverlap[0] > childOverlap[1])
							childOverlap[0] = childOverlap[1];
					}
				}
				__syncthreads();

				// none of the branches overlapped the query
				if( childOverlap[0] == ( NODECARD+1)) {
					passed_mortonCode = node_ptr->index[node_ptr->count-1];

					if( tid == 0 ) rootCount[bid]++;
					node_ptr = root;
					break;
				}
				// there exists some overlapped node
				else{
						node_ptr = node_ptr->child[ childOverlap[0] ];
					if( tid == 0 )
					{
							count[bid]++;
					}
				}
				__syncthreads();
			}

			while( node_ptr->level == 0 )
			{
				isHit = false;

				if ( tid < node_ptr->count && dev_BVH_Node_SOA_Overlap(&query, node_ptr, tid))
				{
					t_hit[tid]++;
					isHit = true;
				}
				__syncthreads();

				passed_mortonCode = node_ptr->index[node_ptr->count-1];

				//last leaf node

				if ( node_ptr->index[node_ptr->count-1]== last_mortonCode )
					break;
				else if( isHit )
				{
					if( tid == 0 ) count[bid]++;
					node_ptr = node_ptr->sibling;
				}
				else
				{
					node_ptr = node_ptr->parent;
					if( tid == 0 )
					{
						if( node_ptr == root)
							rootCount[bid]++;
						else
							count[bid]++;
					}
				}
				__syncthreads();
			}
		}
	}
	__syncthreads();
	int N = NODECARD/2 + NODECARD%2;
	while(N > 1){
		if ( tid < N )
		{
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
		}

		N = N/2 + N%2;
		__syncthreads();
	}

	if(tid==0) {
		if(N==1) 
			hit[bid] = t_hit[0] + t_hit[1];
		else
			hit[bid] = t_hit[0];
	}

	__syncthreads();
}

/*
__global__ void globalMPHR2_BVH(struct Rect* _query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED)
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
	__shared__ int childOverlap[NODECARD];
	__shared__ bool isHit;
	__shared__  struct Rect query;

	t_hit[tid] = 0;
	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	BVH_Node_SOA* root = (BVH_Node_SOA*)deviceBVHRoot[partition_index];


	BVH_Node_SOA* leafNode_ptr = root;
	while( leafNode_ptr->level > 0 )
		leafNode_ptr = leafNode_ptr->child[0];

	int passed_mortonCode;
	int last_mortonCode;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_mortonCode = 0;
		last_mortonCode 	= root->index[root->count-1];

		BVH_Node_SOA* node_ptr 			= root;
		if( tid == 0 ) rootCount[bid]++;
		__syncthreads();

		while( passed_mortonCode < last_mortonCode )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

				if( (tid < node_ptr->count) &&
						(node_ptr->index[tid]> passed_mortonCode) &&
						(dev_BVH_Node_SOA_Overlap(&query, node_ptr, tid)))
				{
					childOverlap[tid] = tid;
				}
				else
				{
					childOverlap[tid] = NODECARD+1;
				}
				__syncthreads();


				// check if I am the leftmost
				// Gather the Overlap idex and compare

				int N = NODECARD/2 + NODECARD%2;
				while(N > 1){
					if ( tid < N ){
						if(childOverlap[tid] > childOverlap[tid+N] )  
							childOverlap[tid] = childOverlap[tid+N];
					}
					N = N/2+N%2;
					__syncthreads();
				}
				if( tid == 0) {
					if(N==1){
						if(childOverlap[0] > childOverlap[1])
							childOverlap[0] = childOverlap[1];
					}
				}
				__syncthreads();

				// none of the branches overlapped the query
				if( childOverlap[0] == ( NODECARD+1)) {
					passed_mortonCode = node_ptr->index[node_ptr->count-1];

					if( tid == 0 ) rootCount[bid]++;
					node_ptr = root;
					break;
				}
				// there exists some overlapped node
				else{
					if( tid == 0 ) count[bid]++;
					node_ptr = node_ptr->child[ childOverlap[0] ];
				}
				__syncthreads();
			}


			while( node_ptr->level == 0 )
			{
				isHit = false;

				if ( tid < node_ptr->count && dev_BVH_Node_SOA_Overlap(&query, node_ptr, tid))
				{
					t_hit[tid]++;
					isHit = true;
				}
				__syncthreads();

				passed_mortonCode = node_ptr->index[node_ptr->count-1];

				//last leaf node

				if ( node_ptr->index[node_ptr->count-1]== last_mortonCode )
					break;
				else if( isHit )
				{
					if( tid == 0 ) count[bid]++;
					node_ptr = node_ptr->sibling;
				}
				else
				{
					node_ptr = node_ptr->parent;
					if( tid == 0 )
					{
						if( node_ptr == root)
							rootCount[bid]++;
						else
							count[bid]++;
					}
				}

				__syncthreads();
			}
		}
	}
	__syncthreads();
	int N = NODECARD/2 + NODECARD%2;
	while(N > 1){
		if ( tid < N )
		{
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
		}

		N = N/2 + N%2;
		__syncthreads();
	}

	if(tid==0) {
		if(N==1) 
			hit[bid] = t_hit[0] + t_hit[1];
		else
			hit[bid] = t_hit[0];
	}

	__syncthreads();

}
*/


__global__ void globalMPHR2_ILP_BVH(struct Rect* _query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED)
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
	__shared__ int childOverlap[NUMTHREADS];
	__shared__ bool isHit;
	__shared__  struct Rect query;

	for ( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS )
		t_hit[thread] = 0;

	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	BVH_Node_SOA* root = (BVH_Node_SOA*)deviceBVHRoot[partition_index];
	int passed_mortonCode;
	int last_mortonCode;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_mortonCode = 0;
		last_mortonCode 	= root->index[root->count-1];

		BVH_Node_SOA* node_ptr 			= root;
		if( tid == 0 ) rootCount[bid]++;

		__syncthreads();
		while( passed_mortonCode < last_mortonCode )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

				__shared__ bool isOverlap;
				isOverlap = false;
				for ( int thread = tid ; thread < NODECARD ; thread+=NUMTHREADS){
					if(
							(thread < node_ptr->count) &&
							(node_ptr->index[thread] > passed_mortonCode) &&
							(dev_BVH_Node_SOA_Overlap(&query, node_ptr, thread)))
					{
						childOverlap[tid] = thread;
						isOverlap = true;
					}
					else
					{
						childOverlap[tid] = NODECARD+1;
					}
					__syncthreads();
					if( isOverlap )
						break;
				}
				__syncthreads();


				// check if I am the leftmost
				// Gather the Overlap idex and compare

				int N = NUMTHREADS/2 + NUMTHREADS%2;
				while(N > 1){
					if( tid < N ){
						if(childOverlap[tid] > childOverlap[tid+N] )  
							childOverlap[tid] = childOverlap[tid+N];
					}
					N = N/2+N%2;
					__syncthreads();
				}
				if( tid == 0) {
					if(N==1){
						if(childOverlap[0] > childOverlap[1]) {
							childOverlap[0] = childOverlap[1];
						}
					}
				}
				__syncthreads();

				// none of the branches overlapped the query
				if( childOverlap[0] == ( NODECARD+1)) {
					passed_mortonCode = node_ptr->index[node_ptr->count-1];

					if( tid == 0 ) rootCount[bid]++;
					node_ptr = root;
					break;
				}
				// there exists some overlapped node
				else{
					node_ptr = node_ptr->child[ childOverlap[0] ];
					if( tid == 0 ) count[bid]++;
				}
				__syncthreads();
			}


			while( node_ptr->level == 0 )
			{
				isHit = false;


				for ( int thread = tid ; thread < node_ptr->count; thread+= NUMTHREADS )
				{
					if ( dev_BVH_Node_SOA_Overlap(&query, node_ptr, thread))
					{
						t_hit[thread]++;
						isHit = true;
					}
				}
				__syncthreads();

				passed_mortonCode = node_ptr->index[node_ptr->count-1];

				//last leaf node

				if ( node_ptr->index[node_ptr->count-1]== last_mortonCode )
					break;
				else if( isHit )
				{
					node_ptr = node_ptr->sibling;
					if( tid == 0 ) count[bid]++;
				}
				else
				{
					node_ptr = node_ptr->parent;
					if( tid == 0 )
					{
						if( node_ptr == root)
							rootCount[bid]++;
						else
							count[bid]++;
					}
				}

				__syncthreads();
			}
		}
	}
	__syncthreads();
	int N = NODECARD/2 + NODECARD%2;
	while(N > 1){
		for ( int thread = tid ; thread < N ; thread+= NUMTHREADS){
			t_hit[thread] = t_hit[thread] + t_hit[thread+N];
		}

		N = N/2 + N%2;
		__syncthreads();
	}

	if(tid==0) {
		if(N==1) 
			hit[bid] = t_hit[0] + t_hit[1];
		else
			hit[bid] = t_hit[0];
	}

	__syncthreads();

}

__global__ void globalMPHR2_TP(struct Rect* _query, int * hit, int* count , long* _leafNode_offset, long* _extendLeafNode_offset , int* rootCount, int NSEARCH, int boundary_index, int PARTITIONED)
{

	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int query_id = ( blockIdx.x*blockDim.x) + threadIdx.x;

	int tree_index=0;

	int partition_index=0;
	int block_init_position = query_id;
	int block_incremental_value = NSEARCH;


	const long leafNode_offset = _leafNode_offset[tree_index];
	const long extendLeafNode_offset = _extendLeafNode_offset[tree_index];

	__shared__ int t_hit[NUM_TP]; 
	__shared__ int t_rootCount[NUM_TP]; 
	__shared__ int t_count[NUM_TP]; 

	struct Rect query;
	struct Node_SOA* node_ptr;


	t_hit[tid] = 0;
	t_rootCount[tid] = 0;
	t_count[tid] = 0;

	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	struct Node_SOA* root = (struct Node_SOA*) deviceRoot[partition_index];
	struct Node_SOA* leafNode_ptr = (struct Node_SOA*) ( (char*) root+(PGSIZE*leafNode_offset) );
	struct Node_SOA* extendNode_ptr = (struct Node_SOA*) ( (char*) root+(PGSIZE*extendLeafNode_offset) );

	int passed_hIndex;
	int last_hIndex;

	for( int n = block_init_position; n < NSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_hIndex = 0;
		last_hIndex 	= root->index[root->count-1];

		node_ptr 			= root;
		t_rootCount[tid]++;

		while( passed_hIndex < last_hIndex )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

				int t;
				for( t=0; t < node_ptr->count; t++)
				{
					if ((node_ptr->index[t]> passed_hIndex) &&
							(dev_Node_SOA_Overlap(&query, node_ptr, t)))
					{
						break;
					}
				}


				// none of the branches overlapped the query
				if( t ==  node_ptr->count )
				{
					passed_hIndex = node_ptr->index[node_ptr->count-1];

					node_ptr = root;
					t_rootCount[tid]++; // would be problem?? abt sync..
					break;
				}
				 //there exists some overlapped node
				else{
					node_ptr = node_ptr->child[ t ];
					t_count[tid]++;
				}
			}

			while( node_ptr->level == 0 )
			{

				bool isHit;
				isHit = false;

				int t;
				for( t = 0; t < node_ptr->count; t++)
				{
					if ( dev_Node_SOA_Overlap(&query, node_ptr, t))
					{
						t_hit[tid]++;
						isHit = true;
					}
				}

				passed_hIndex = node_ptr->index[node_ptr->count-1];

				//last leaf node

				if ( node_ptr->index[node_ptr->count-1] == last_hIndex )
					break;
				else if( isHit )
				{
					node_ptr++;
					t_count[tid]++;
				}
				else
				{
					node_ptr = extendNode_ptr + ( ( node_ptr - leafNode_ptr) / NODECARD) ;
					if( node_ptr == root)
						t_rootCount[tid]++;
					else
						t_count[tid]++;
				}
			}

		}
	}

	__syncthreads();
	int N = NUM_TP/2 + NUM_TP%2;
	while(N > 1){
		if ( tid < N )
		{
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
			t_rootCount[tid] = t_rootCount[tid] + t_rootCount[tid+N];
			t_count[tid] = t_count[tid] + t_count[tid+N];
		}

		N = N/2 + N%2;
		__syncthreads();
	}

	if(tid==0) {
		if(N==1) 
		{
			hit[bid] = t_hit[0] + t_hit[1];
			rootCount[bid] = t_rootCount[0] + t_rootCount[1];
			count[bid] = t_count[0] + t_count[1];
		}
		else
		{
			hit[bid] = t_hit[0];
			rootCount[bid] = t_rootCount[0];
			count[bid] = t_count[0];
		}
	}
}

__global__ void globalMPHR2_BVH_TP(struct Rect* _query, int * hit, int* count , int* rootCount, int NSEARCH, int PARTITIONED  )
{

	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int query_id = ( blockIdx.x*blockDim.x) + threadIdx.x;

	int partition_index=0;
	int block_init_position = query_id;
	int block_incremental_value = NSEARCH;

	__shared__ int t_hit[NUM_TP]; 
	__shared__ int t_rootCount[NUM_TP]; 
	__shared__ int t_count[NUM_TP]; 

	struct Rect query;
	BVH_Node_SOA* node_ptr;


	t_hit[tid] = 0;
	t_rootCount[tid] = 0;
	t_count[tid] = 0;

	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	BVH_Node_SOA* root = (BVH_Node_SOA*) deviceBVHRoot[partition_index];

	int passed_Index;
	int last_Index;

	for( int n = block_init_position; n < NSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_Index = 0;
		last_Index 	= root->index[root->count-1];

		node_ptr 			= root;
		t_rootCount[tid]++;

		while( passed_Index < last_Index )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

				int t;
				for( t=0; t < node_ptr->count; t++)
				{
					if ((node_ptr->index[t]> passed_Index) &&
							(dev_BVH_Node_SOA_Overlap(&query, node_ptr, t)))
					{
						break;
					}
				}


				// none of the branches overlapped the query
				if( t ==  node_ptr->count )
				{
					passed_Index = node_ptr->index[node_ptr->count-1];

					node_ptr = root;
					t_rootCount[tid]++; // would be problem?? abt sync..
					break;
				}
				 //there exists some overlapped node
				else{
					node_ptr = node_ptr->child[ t ];
					t_count[tid]++;
				}
			}

			while( node_ptr->level == 0 )
			{

				bool isHit;
				isHit = false;

				int t;
				for( t = 0; t < node_ptr->count; t++)
				{
					if ( dev_BVH_Node_SOA_Overlap(&query, node_ptr, t))
					{
						t_hit[tid]++;
						isHit = true;
					}
				}

				passed_Index = node_ptr->index[node_ptr->count-1];

				//last leaf node

				if ( node_ptr->index[node_ptr->count-1] == last_Index )
					break;
				else if( isHit )
				{
					node_ptr = node_ptr->sibling;
					t_count[tid]++;
				}
				else
				{
					node_ptr = node_ptr->parent;
					if( node_ptr == root)
						t_rootCount[tid]++;
					else
						t_count[tid]++;
				}
			}

		}
	}

	__syncthreads();
	int N = NUM_TP/2 + NUM_TP%2;
	while(N > 1){
		if ( tid < N )
		{
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
			t_rootCount[tid] = t_rootCount[tid] + t_rootCount[tid+N];
			t_count[tid] = t_count[tid] + t_count[tid+N];
		}

		N = N/2 + N%2;
		__syncthreads();
	}

	if(tid==0) {
		if(N==1) 
		{
			hit[bid] = t_hit[0] + t_hit[1];
			rootCount[bid] = t_rootCount[0] + t_rootCount[1];
			count[bid] = t_count[0] + t_count[1];
		}
		else
		{
			hit[bid] = t_hit[0];
			rootCount[bid] = t_rootCount[0];
			count[bid] = t_count[0];
		}
	}
}

