#include <skippointer.h>

//#####################################################################
//####################### IMPLEMENTATIONS #############################
//#####################################################################


void skippointer(int number_of_procs, int myrank)
{
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
	float elapsed_time = 0.0f;

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
	int h_skipCount[NUMBLOCK];

	long t_hit = 0;
	int t_count = 0;
	int t_rootCount = 0;
	int t_skipCount = 0;

	int* d_hit;
	int* d_count;
	int* d_rootCount;
	int* d_skipCount;

	cudaMalloc( (void**) &d_hit,   NUMBLOCK*sizeof(int) );
	cudaMalloc( (void**) &d_count, NUMBLOCK*sizeof(int) );
	cudaMalloc( (void**) &d_rootCount, NUMBLOCK*sizeof(int) );
	cudaMalloc( (void**) &d_skipCount, NUMBLOCK*sizeof(int) );


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
			for(nBatch=0; nBatch < NUMBLOCK*NUM_TP&& i < NUMSEARCH; nBatch++, i++) 
#else
			for(nBatch=0; nBatch < NUMBLOCK && i < NUMSEARCH; nBatch++, i++) 
#endif
			{
				//DEBUG
				for(int j=0;j<2*NUMDIMS;j++){
					fread(&query[nBatch].boundary[j], sizeof(float), 1, fp);
				}	
			}
			cudaMemcpy(d_query, query, nBatch*sizeof(struct Rect), cudaMemcpyHostToDevice);

#ifdef ILP
				globalSkippointer_ILP_BVH<<<nBatch, NUMTHREADS>>>(d_query, d_hit, nBatch, PARTITIONED );
#else
#ifdef TP
				globalSkippointer_BVH_TP<<<NUMBLOCK, NUM_TP>>>(d_query, d_hit, d_count, d_rootCount, d_skipCount, nBatch, PARTITIONED );
#else
				globalSkippointer_BVH<<<nBatch, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, d_skipCount, nBatch, PARTITIONED );
#endif
#endif

#ifdef TP
			cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_skipCount, d_skipCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<NUMBLOCK;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
				t_rootCount += h_rootCount[j]; 
				t_skipCount += h_skipCount[j]; 
			}
#else
			cudaMemcpy(h_hit, d_hit, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_skipCount, d_skipCount, nBatch*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<nBatch;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
				t_rootCount += h_rootCount[j]; 
				t_skipCount += h_skipCount[j]; 
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
				globalSkippointer_BVH<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, d_skipCount, 1, PARTITIONED );
#else
				globalSkippointer_ILP_BVH<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, 1, PARTITIONED );
#endif

			cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_skipCount, d_skipCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<NUMBLOCK;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
				t_rootCount += h_rootCount[j]; 
				t_skipCount += h_skipCount[j]; 
			}
		}
		cudaThreadSynchronize();
		cudaEventRecord(stop_event, 0) ;
		cudaEventSynchronize(stop_event) ;

		cudaFree( d_query );
	}


	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	printf("Skip pointer time          : %.3f ms\n", elapsed_time);
	printf("Skip pointer HIT           : %lu\n",t_hit);
#ifndef ILP
	cout<< "Skip pointer visited       : " << (t_count)/(float)NUMSEARCH << endl;
	cout<< "Skip pointer root visited  : " << (t_rootCount)/(float)NUMSEARCH << endl;
	cout<< "Skip pointer skip Count    : " << (t_skipCount)/(float)NUMSEARCH << endl;
#endif
	printf("Skip pointer HIT ratio(%)  : %5.5f\n",((t_hit/NUMSEARCH)/(float)(NUMDATA/100)));

	t_time[6] = elapsed_time;
	t_visit[6] = (t_count)/(float)NUMSEARCH;
	t_rootVisit[6] = (t_rootCount)/(float)NUMSEARCH;
	t_skipPointer[6] = (t_skipCount)/(float)NUMSEARCH;

	fclose(fp);

	cudaFree( d_hit );
	cudaFree( d_count );
	cudaFree( d_rootCount );

}

__global__ void globalSkippointer_BVH (struct Rect* _query, int * hit, int* count , int* rootCount, int* skipCount, int mpiSEARCH, int PARTITIONED)
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
	__shared__ struct Rect query;

	t_hit[tid] = 0;
	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;
	skipCount[bid] = 0;

	BVH_Node_SOA* root = (BVH_Node_SOA*)deviceBVHRoot[partition_index];


	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		BVH_Node_SOA* node = root;
		BVH_Node_SOA* stopNode = root->sibling;

		if( tid  == 0 ) rootCount[bid] ++;

		while(node != stopNode)
		{
			if( node->level > 0 )
			{

				if( (tid < node->count) &&
						(dev_BVH_Node_SOA_Overlap(&query, node, tid)))
				{
					childOverlap[tid] = tid;
				}
				else
				{
					childOverlap[tid] = NODECARD+1;
				}
				__syncthreads();

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
				if( childOverlap[0] == ( NODECARD+1)) 
				{
					//sibling => skip pointer 
					node = node->sibling;
//					if( tid == 0)
						skipCount[bid]++;
//					__syncthreads();
				}
				else //overlap child
				{
					node++;

					if( tid == 0)
					{
						if( node == root) rootCount[bid] ++;
						else  						count[bid]++;
					}
					__syncthreads();
				}
			}
			
			if( node->level == 0)
			{
				if ( (tid < node->count) && (dev_BVH_Node_SOA_Overlap(&query, node, tid)))
				{
					t_hit[tid]++;
				}

				node++;

				if( tid == 0)
				{
					if( node == root) rootCount[bid] ++;
					else  						count[bid]++;
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
__global__ void globalSkippointer_ILP_BVH(struct Rect* _query, int * hit, int mpiSEARCH, int PARTITIONED)
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
	__shared__ struct Rect query;

	for ( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS)
		t_hit[thread] = 0;

	t_hit[tid] = 0;
	hit[bid] = 0;

	BVH_Node_SOA* root = (BVH_Node_SOA*)deviceBVHRoot[partition_index];


	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		BVH_Node_SOA* node = root;
		BVH_Node_SOA* stopNode = root->sibling;

		
		while(node != stopNode)
		{
			if( node->level > 0 )
			{
				for ( int thread = tid ; thread < NODECARD ; thread+=NUMTHREADS){
					if( (thread < node->count) &&
							(dev_BVH_Node_SOA_Overlap(&query, node, thread)))
					{
						childOverlap[thread] = thread;
					}else{
						childOverlap[thread] = NODECARD+1;
					}
				}
				__syncthreads();

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
				
				//No overlap child
				if( childOverlap[0] == ( NODECARD+1))
				{
					//sibling is skip poniter
					node = node->sibling;
				}
				else //overlap child
				{
					node++;
				}
			}

			if(node->level == 0)
			{
				for ( int thread = tid ; thread < node->count ; thread+=NUMTHREADS){
					if ( dev_BVH_Node_SOA_Overlap(&query, node, thread ))
					{
						t_hit[thread]++;
					}
				}
				__syncthreads();
				node++;

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
__global__ void globalSkippointer_BVH_TP (struct Rect* _query, int * hit, int* count , int* rootCount, int* skipCount, int mpiSEARCH, int PARTITIONED)
{

	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int query_id = ( blockIdx.x*blockDim.x) + threadIdx.x;

	int partition_index=0;
	int block_init_position = query_id;
	int block_incremental_value = mpiSEARCH;

	__shared__ int t_hit[NUM_TP]; 
	__shared__ int t_rootCount[NUM_TP]; 
	__shared__ int t_count[NUM_TP]; 
	__shared__ int t_skipCount[NUM_TP]; 

	struct Rect query;

	t_hit[tid] = 0;
	t_rootCount[tid] = 0;
	t_count[tid] = 0;
	t_skipCount[tid] = 0;

	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;
	skipCount[bid] = 0;

	BVH_Node_SOA* root = (BVH_Node_SOA*)deviceBVHRoot[partition_index];


	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		BVH_Node_SOA* node = root;
		BVH_Node_SOA* stopNode = root->sibling;

		t_rootCount[tid] ++;

		while(node != stopNode)
		{
			if( node->level > 0 )
			{
				int t;
				for( t=0; t < node->count; t++)
				{
					if (dev_BVH_Node_SOA_Overlap(&query, node, t))
					{
						break;
					}
				}

				//No overlap child
				if( t == node->count )
				{
					//sibling => skip pointer 
					node = node->sibling;
					t_skipCount[tid]++;
				}
				else //overlap child
				{
					node++;

					if( tid == 0)
					{
						if( node == root) t_rootCount[tid] ++;
						else  						t_count[tid]++;
					}
				}
			}
			
			if( node->level == 0)
			{
				int t;
				for( t = 0; t < node->count; t++)
				{
					if ( dev_BVH_Node_SOA_Overlap(&query, node, t))
					{
						t_hit[tid]++;
					}
				}

				node++;

					if( node == root) t_rootCount[tid] ++;
					else  						t_count[tid]++;
			}
		}
	}

	__syncthreads();
	int N = NUM_TP/2 + NUM_TP%2;
	while(N > 1){
		if ( tid < N ){
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
			t_rootCount[tid] = t_rootCount[tid] + t_rootCount[tid+N];
			t_count[tid] = t_count[tid] + t_count[tid+N];
			t_skipCount[tid] = t_skipCount[tid] + t_skipCount[tid+N];
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
			skipCount[bid] = t_skipCount[0] + t_skipCount[1];
		}
		else
		{
			hit[bid] = t_hit[0];
			rootCount[bid] = t_rootCount[0];
			count[bid] = t_count[0];
			skipCount[bid] = t_skipCount[0];
		}
	}

	
	__syncthreads();

}
