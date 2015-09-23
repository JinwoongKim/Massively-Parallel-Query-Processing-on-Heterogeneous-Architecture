#include <shortstack.h>
//#####################################################################
//####################### IMPLEMENTATIONS #############################
//#####################################################################


void shortstack(int number_of_procs, int myrank)
{
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
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
	int h_pushCount[NUMBLOCK];
	int h_popCount[NUMBLOCK];

	long t_hit = 0;
	int t_count = 0;
	int t_rootCount = 0;
	int t_pushCount = 0;
	int t_popCount = 0;

	int* d_hit;
	int* d_count;
	int* d_rootCount;
	int* d_pushCount;
	int* d_popCount;

	cudaMalloc( (void**) &d_hit,   NUMBLOCK*sizeof(int) );
	cudaMalloc( (void**) &d_count, NUMBLOCK*sizeof(int) );
	cudaMalloc( (void**) &d_rootCount, NUMBLOCK*sizeof(int) );
	cudaMalloc( (void**) &d_pushCount, NUMBLOCK*sizeof(int) );
	cudaMalloc( (void**) &d_popCount, NUMBLOCK*sizeof(int) );


	if( PARTITIONED == 1 )
	{
		struct Rect* d_query;
#ifdef TP
		cudaMalloc( (void**) &d_query, NUMBLOCK*NODECARD*sizeof(struct Rect) );
		struct Rect query[NUMBLOCK*NODECARD];
#else
		cudaMalloc( (void**) &d_query, NUMBLOCK*sizeof(struct Rect) );
		struct Rect query[NUMBLOCK];
#endif


		cudaEventRecord(start_event, 0);

		for( int i = 0 ; i < NUMSEARCH; )
		{
			int nBatch=0;
#ifdef TP
			for(nBatch=0; nBatch < NUMBLOCK*NODECARD && i < NUMSEARCH; nBatch++, i++) 
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

			if( BUILD_TYPE == 0)
				globalShortstack_ILP<<<nBatch,NUMTHREADS>>>(d_query, d_hit, nBatch, PARTITIONED );
			else
				globalShortstack_ILP_BVH<<<nBatch, NUMTHREADS>>>(d_query, d_hit, nBatch, PARTITIONED );
#else
#ifdef TP
			if( BUILD_TYPE == 0)
				globalShortstack_TP<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, d_pushCount, d_popCount, nBatch, PARTITIONED );
			else
				globalShortstack_BVH_TP<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, d_pushCount, d_popCount, nBatch, PARTITIONED );
#else
			if( BUILD_TYPE == 0)
				globalShortstack<<<nBatch, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, d_pushCount, d_popCount, nBatch, PARTITIONED );
			else
				globalShortstack_BVH<<<nBatch, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, d_pushCount, d_popCount, nBatch, PARTITIONED );
#endif
#endif

#ifdef TP
			cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_pushCount, d_pushCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_popCount, d_popCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<NUMBLOCK;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
				t_rootCount += h_rootCount[j]; 
				t_pushCount += h_pushCount[j];
				t_popCount += h_popCount[j]; 
			}
		}
#else
			cudaMemcpy(h_hit, d_hit, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_pushCount, d_pushCount, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_popCount, d_popCount, nBatch*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<nBatch;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
				t_rootCount += h_rootCount[j]; 
				t_pushCount += h_pushCount[j];
				t_popCount += h_popCount[j]; 
			}
	  }
#endif

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
				globalShortstack<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, d_pushCount, d_popCount, 1, PARTITIONED );
			else
				globalShortstack_BVH<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, d_pushCount, d_popCount, 1, PARTITIONED );
#else
			if( BUILD_TYPE == 0)
				globalShortstack_ILP<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, 1, PARTITIONED );
			else
				globalShortstack_ILP_BVH<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, 1, PARTITIONED );
#endif

			cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_pushCount, d_pushCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_popCount, d_popCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<NUMBLOCK;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
				t_rootCount += h_rootCount[j]; 
				t_pushCount += h_pushCount[j];
				t_popCount += h_popCount[j]; 
			}
		}
		cudaThreadSynchronize();
		cudaEventRecord(stop_event, 0) ;
		cudaEventSynchronize(stop_event) ;

		cudaFree( d_query );
	}


	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	printf("Short Stack time          : %.3f ms\n", elapsed_time);
	printf("Short Stack HIT           : %lu\n",t_hit);
#ifndef ILP
	printf("Short Stack visited       : %.6f\n", (t_count)/(float)NUMSEARCH ) ;
	printf("Short Stack root visited  : %.6f\n", (t_rootCount)/(float)NUMSEARCH ) ;
	printf("Short Stack pushCount     : %.6f\n", (t_pushCount)/(float)NUMSEARCH ) ;
	printf("Short Stack popCount      : %.6f\n", (t_popCount)/(float)NUMSEARCH ) ;
#endif
	printf("Short Stack HIT ratio(%)  : %5.5f\n",((t_hit/NUMSEARCH)/(float)(NUMDATA/100)));

	t_time[4] = elapsed_time;

	t_visit[4] = (t_count)/(float)NUMSEARCH;
	t_rootVisit[4] = (t_rootCount)/(float)NUMSEARCH;
	t_push[4] = (t_pushCount)/(float)NUMSEARCH;
	t_pop[4] = (t_popCount)/(float)NUMSEARCH;

	fclose(fp);

	cudaFree( d_hit );
	cudaFree( d_count );
	cudaFree( d_rootCount );

}
__global__ void globalShortstack(struct Rect* _query, int * hit, int* count , int* rootcount, int* pushCount, int* popCount, int mpiSEARCH, int PARTITIONED)
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

	__shared__ struct Node_SOA stack[STACK_SIZE];
	__shared__ struct Rect query;
	__shared__ struct Node_SOA* node;
	__shared__ int numOfoverlapChilds[NODECARD];
	__shared__ int top;
	__shared__ int size;

	__shared__ int t_hit[NODECARD]; 
	__shared__ int childOverlap[NODECARD];

	if( tid == 0)
	{
		top = -1;
		size = 0;
	}
		__syncthreads();

	t_hit[tid] = 0;
	hit[bid] = 0;
	count[bid] = 0;
	pushCount[bid] = 0;
	popCount[bid] = 0;
	rootcount[bid] = 0;

	__syncthreads();

	struct Node_SOA* root = (struct Node_SOA*)deviceRoot[partition_index];

	int passed_hIndex;
	int last_hIndex;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];

		passed_hIndex = 0;
		last_hIndex 	= root->index[root->count-1];


		if( tid == 0)
		{
			node = root;
			rootcount[bid]++;
		}
		__syncthreads();

		while( passed_hIndex < last_hIndex )
		{	
			while( node->level > 0 ) {
				numOfoverlapChilds[tid] = 0;

				if( (tid < node->count) &&
						(node->index[tid] > passed_hIndex) &&
						(dev_Node_SOA_Overlap(&query, node, tid)))
				{
					numOfoverlapChilds[tid] = 1;
//					atomicAdd(&numOfoverlapChilds,1);
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

						numOfoverlapChilds[tid] += numOfoverlapChilds[tid+N];
					}

					N = N/2+N%2;
					__syncthreads();
				}
				if( tid == 0) {
					if(N==1){
						if(childOverlap[0] > childOverlap[1])
							childOverlap[0] = childOverlap[1];

						numOfoverlapChilds[0] += numOfoverlapChilds[1];
					}
				}
				__syncthreads();

				// none of the branches overlapped the query
				if( childOverlap[0] == ( NODECARD+1)) {
					passed_hIndex = node->index[node->count-1];

					//stack.pop()
					if( size > 0 )
					{

						if( tid == 0 )
						{
								node = &stack[top];
								if( top == 0) top = STACK_SIZE-1;
  							else					top--;

								popCount[bid]++;
								
								size--;
						}
						__syncthreads();
					}
					else
					{
						if( tid == 0)
						{
							node = root;
							rootcount[bid]++;
						}
					}
					__syncthreads();
					break;
				} // there exists some overlapped node
				else{

					if( numOfoverlapChilds[0] > 1 )
					{

						//stack.push()
						if( tid == 0 )
						{
							if( top == STACK_SIZE-1) top = 0;
							else										 ++top;

						if( size < STACK_SIZE-1) size++;
						}
						__syncthreads();

						if( node != &stack[top])
						{
							__shared__ char* shared_ptr;
							__shared__ char* global_ptr;

							if( tid == 0)
							{
								pushCount[bid]++;

								shared_ptr = (char*) &stack[top];
								global_ptr = (char*) node;

								memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*12), global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*12), sizeof(int)*2);
							}
							__syncthreads();

							for( int d=0; d<NUMDIMS*2; d++)
								memcpy(shared_ptr+(NODECARD*4)*d+(tid*4), global_ptr+(NODECARD*4)*d+(tid*4), sizeof(float));
							memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+tid*4, global_ptr+(8*NODECARD*NUMDIMS)+tid*4, sizeof(int)); // index
							memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+tid*4, global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+tid*4, sizeof(int));  // half of child pointers
							memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*8)+tid*4, global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*8)+tid*4, sizeof(int));  // the other child pointers
							__syncthreads();
						}
					}
					if( tid == 0)
						{
							node = node->child[ childOverlap[0] ];
							count[bid]++;
						}
				}
				__syncthreads();
			}

			if( node->level == 0 )
			{
				if ( tid < node->count
						&& dev_Node_SOA_Overlap(&query, node, tid))
				{
					t_hit[tid]++;
				}
				__syncthreads();

				passed_hIndex = node->index[node->count-1];

				//stack.pop()
				if( size > 0 )
				{
					if( tid == 0 )
					{
						node = &stack[top];

						popCount[bid]++;

						if( top == 0) top = STACK_SIZE-1;
						else					top--;

						size--;
					}
					__syncthreads();
				}
				else
				{
					if( tid == 0)
					{
						node = root;
						rootcount[bid]++;
					}
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
__global__ void globalShortstack_BVH (struct Rect* _query, int * hit, int* count , int* rootCount, int* pushCount, int* popCount, int mpiSEARCH, int PARTITIONED)
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

	__shared__ BVH_Node_SOA stack[STACK_SIZE];
	__shared__ struct Rect query; 
	__shared__ BVH_Node_SOA* node;
	__shared__ int numOfoverlapChilds[NODECARD];
	__shared__ int top;
	__shared__ int size;

	__shared__ int t_hit[NODECARD]; 
	__shared__ int childOverlap[NODECARD];

	if( tid == 0)
	{
		top = -1;
		size = 0;
	}
		__syncthreads();

	t_hit[tid] = 0;
	hit[bid] = 0;
	count[bid] = 0;
	pushCount[bid] = 0;
	popCount[bid] = 0;
	rootCount[bid] = 0;

	__syncthreads();

	BVH_Node_SOA* root = (BVH_Node_SOA*)deviceBVHRoot[partition_index];

	int passed_mortonCode;
	int last_mortonCode;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];

		passed_mortonCode = 0;
		last_mortonCode 	= root->index[root->count-1];

		if( tid == 0)
		{
			node = root;
			rootCount[bid]++;
		}
		__syncthreads();

		while( passed_mortonCode < last_mortonCode )
		{	
			while( node->level > 0 ) {

				numOfoverlapChilds[tid] = 0;

				if( (tid < node->count) &&
						(node->index[tid]> passed_mortonCode) &&
						(dev_BVH_Node_SOA_Overlap(&query, node, tid)))
				{
					numOfoverlapChilds[tid]=1;
					//atomicAdd(&numOfoverlapChilds, 1);
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

						numOfoverlapChilds[tid] += numOfoverlapChilds[tid+N];
					}

					N = N/2+N%2;
					__syncthreads();
				}
				if( tid == 0) {
					if(N==1){
						if(childOverlap[0] > childOverlap[1])
							childOverlap[0] = childOverlap[1];
						numOfoverlapChilds[0] += numOfoverlapChilds[1];
					}
				}
				__syncthreads();

				// none of the branches overlapped the query
				if( childOverlap[0] == ( NODECARD+1)) {
					passed_mortonCode = node->index[node->count-1];

					//stack.pop()
					if( size > 0 )
					{
						if( tid == 0 )
						{
								node = &stack[top];
								if( top == 0) top = STACK_SIZE-1;
								else					top--;
								
								size--;
								popCount[bid]++;
						}
						__syncthreads();
					}
					else
					{
						if( tid == 0)
						{
							node = root;
							rootCount[bid]++;
						}
					}
					__syncthreads();

					break;
				} // there exists some overlapped node
				else{

					if( numOfoverlapChilds[0] > 1 )
					{
						//stack.push()

						if( tid == 0 )
						{
							if( top == STACK_SIZE-1) top = 0;
							else										 ++top;

							if( size < STACK_SIZE-1) size++;
						}
						__syncthreads();

						if( node != &stack[top])
						{
							__shared__ char* shared_ptr;
							__shared__ char* global_ptr;

							if( tid == 0)
							{
								pushCount[bid]++;

								shared_ptr = (char*) &stack[top];
								global_ptr = (char*) node;

								memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*12), global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*12), (sizeof(int)*6));
							}
							__syncthreads();

							for( int d=0; d<NUMDIMS*2; d++)
								memcpy(shared_ptr+(NODECARD*4)*d+(tid*4), global_ptr+(NODECARD*4)*d+(tid*4), sizeof(float));
							memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+tid*4, global_ptr+(8*NODECARD*NUMDIMS)+tid*4, sizeof(int)); // index
							memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+tid*4, global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+tid*4, sizeof(int));  // half of child pointers
							memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*8)+tid*4, global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*8)+tid*4, sizeof(int));  // the other child pointers
							__syncthreads();

						}
					}


					if( tid == 0)
					{
						node = node->child[ childOverlap[0] ];
						count[bid]++;
					}
					//if there exists child more than one, stack.push
				}
				__syncthreads();
			}

			if( node->level == 0 )
			{

				if ( tid < node->count
						&& dev_BVH_Node_SOA_Overlap(&query, node, tid))
				{
					t_hit[tid]++;
				}
				__syncthreads();

				passed_mortonCode = node->index[node->count-1];

				//stack.pop()
				if( size > 0 )
				{
					if( tid == 0 )
					{
						node = &stack[top];

						popCount[bid]++;

						if( top == 0) top = STACK_SIZE-1;
						else					top--;

						size--;
					}
					__syncthreads();
				}
				else
				{
					if( tid == 0)
					{
						node = root;
						rootCount[bid]++;
					}
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
__global__ void globalShortstack_ILP(struct Rect* _query, int * hit, int mpiSEARCH, int PARTITIONED  )

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

	__shared__ struct Node_SOA stack[STACK_SIZE];
	__shared__ struct Rect query; // should be shared mem.
	__shared__ struct Node_SOA* node;
	__shared__ int numOfoverlapChilds;
	__shared__ int top;
	__shared__ int size;

	__shared__ int t_hit[NODECARD]; 
	__shared__ int childOverlap[NODECARD];

	if( tid == 0)
	{
		top = -1;
		size = 0;
	}
		__syncthreads();

	for(int thread = tid ; thread < NODECARD; thread+=NUMTHREADS)
		t_hit[thread] = 0;

	hit[bid] = 0;

	__syncthreads();

	struct Node_SOA* root = (struct Node_SOA*)deviceRoot[partition_index];

	int passed_hIndex;
	int last_hIndex;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];

		passed_hIndex = 0;
		last_hIndex 	= root->index[root->count-1];


		if( tid == 0)
		{
			node = root;
		}
		__syncthreads();

		while( passed_hIndex < last_hIndex )
		{	
			while( node->level > 0 ) {

				if( tid == 0 )
					numOfoverlapChilds = 0;
				__syncthreads();

				for ( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS)
				{
					if( (thread < node->count ) &&
							(node->index[thread] > passed_hIndex) &&
							(dev_Node_SOA_Overlap(&query, node, thread)))
					{
						atomicAdd(&numOfoverlapChilds, 1);
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
					passed_hIndex = node->index[node->count-1];

					//stack.pop()
					if( size > 0 )
					{

						if( tid == 0 )
						{
								node = &stack[top];
								if( top == 0) top = STACK_SIZE-1;
								else					top--;
								
								size--;
						}
						__syncthreads();
					}
					else
					{
						if( tid == 0)
						{
							node = root;
						}
					}
					__syncthreads();

					break;
				} // there exists some overlapped node
				else{

					if( numOfoverlapChilds > 1 )
					{
						//stack.push()

						if( tid == 0 )
						{
							if( top == STACK_SIZE-1) top = 0;
							else										 ++top;
							if( size < STACK_SIZE-1) size++;
						}
						__syncthreads();

						if( node != &stack[top])
						{
							__shared__ char* shared_ptr;
							__shared__ char* global_ptr;

							if( tid == 0)
							{
								shared_ptr = (char*) &stack[top];
								global_ptr = (char*) node;

								memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*12), global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*12), sizeof(int)*2);
							}
							__syncthreads();


							for ( int thread = tid ; thread < NODECARD ; thread+=NUMTHREADS)
							{
								for( int d=0; d<NUMDIMS*2; d++)
									memcpy(shared_ptr+(NODECARD*4)*d+(thread*4), global_ptr+(NODECARD*4)*d+(thread*4), sizeof(float));
								memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+thread*4, global_ptr+(8*NODECARD*NUMDIMS)+thread*4, sizeof(int)); // index
								memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+thread*4, global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+thread*4, sizeof(int));  // half of child pointers
								memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*8)+thread*4, global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*8)+thread*4, sizeof(int));  // the other child pointers
							}
							__syncthreads();
						}
					}

					if( tid == 0)
					{
						node = node->child[ childOverlap[0] ];
					}
				}
				__syncthreads();
			}

			if( node->level == 0 )
			{

				for ( int thread = tid ; thread < NODECARD ; thread+=NUMTHREADS){
					if ( thread < node->count && 
							 dev_Node_SOA_Overlap(&query, node, thread) )
					{
						t_hit[thread]++;
					}
				}
				__syncthreads();

				passed_hIndex = node->index[node->count-1];

				//stack.pop()
				if( size > 0 )
				{
					if( tid == 0 )
					{
						node = &stack[top];
						if( top == 0) top = STACK_SIZE-1;
						else					top--;

						size--;
					}
					__syncthreads();
				}
				else
				{
					if( tid == 0)
					{
						node = root;
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
}
__global__ void globalShortstack_ILP_BVH(struct Rect* _query, int * hit, int mpiSEARCH, int PARTITIONED)
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

	__shared__ BVH_Node_SOA stack[STACK_SIZE];
	__shared__ struct Rect query; // should be shared mem.
	__shared__ BVH_Node_SOA* node;
	__shared__ int numOfoverlapChilds;
	__shared__ int top;
	__shared__ int size;

	__shared__ int t_hit[NODECARD]; 
	__shared__ int childOverlap[NODECARD];

	if( tid == 0)
	{
		top = -1;
		size = 0;
	}
		__syncthreads();

	for(int thread = tid ; thread < NODECARD; thread+=NUMTHREADS)
		t_hit[thread] = 0;

	hit[bid] = 0;

	__syncthreads();

	BVH_Node_SOA* root = (BVH_Node_SOA*)deviceBVHRoot[partition_index];

	int passed_hIndex;
	int last_hIndex;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];

		passed_hIndex = 0;
		last_hIndex 	= root->index[root->count-1];


		if( tid == 0)
		{
			node = root;
		}
		__syncthreads();

		while( passed_hIndex < last_hIndex )
		{	
			while( node->level > 0 ) {

				if( tid == 0 )
					numOfoverlapChilds = 0;
				__syncthreads();

				for ( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS)
				{
					if( (thread < node->count ) &&
							(node->index[thread] > passed_hIndex) &&
							(dev_BVH_Node_SOA_Overlap(&query, node, thread)))
					{
						atomicAdd(&numOfoverlapChilds, 1);
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
					passed_hIndex = node->index[node->count-1];

					//stack.pop()
					if( size > 0 )
					{

						if( tid == 0 )
						{
								node = &stack[top];
								if( top == 0) top = STACK_SIZE-1;
								else					top--;
								
								size--;
						}
						__syncthreads();
					}
					else
					{
						if( tid == 0)
						{
							node = root;
						}
					}
					__syncthreads();

					break;
				} // there exists some overlapped node
				else{

					if( numOfoverlapChilds > 1 )
					{
						//stack.push()

						if( tid == 0 )
						{
							if( top == STACK_SIZE-1) top = 0;
							else										 ++top;
							if( size < STACK_SIZE-1) size++;
						}
						__syncthreads();

						if( node != &stack[top])
						{
							__shared__ char* shared_ptr;
							__shared__ char* global_ptr;

							if( tid == 0)
							{
								shared_ptr = (char*) &stack[top];
								global_ptr = (char*) node;

								memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*12), global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*12), (sizeof(int)*2)+(sizeof(long)*2));
							}
							__syncthreads();


							for ( int thread = tid ; thread < NODECARD ; thread+=NUMTHREADS)
							{
								for( int d=0; d<NUMDIMS*2; d++)
									memcpy(shared_ptr+(NODECARD*4)*d+(thread*4), global_ptr+(NODECARD*4)*d+(thread*4), sizeof(float));
								memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+thread*4, global_ptr+(8*NODECARD*NUMDIMS)+thread*4, sizeof(int)); // index
								memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+thread*4, global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+thread*4, sizeof(int));  // half of child pointers
								memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*8)+thread*4, global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*8)+thread*4, sizeof(int));  // the other child pointers
							}
							__syncthreads();
						}
					}

					if( tid == 0)
					{
						node = node->child[ childOverlap[0] ];
					}
				}
				__syncthreads();
			}

			if( node->level == 0 )
			{

				for ( int thread = tid ; thread < NODECARD ; thread+=NUMTHREADS){
					if ( thread < node->count && 
							 dev_BVH_Node_SOA_Overlap(&query, node, thread) )
					{
						t_hit[thread]++;
					}
				}
				__syncthreads();

				passed_hIndex = node->index[node->count-1];

				//stack.pop()
				if( size > 0 )
				{
					if( tid == 0 )
					{
						node = &stack[top];
						if( top == 0) top = STACK_SIZE-1;
						else					top--;

						size--;
					}
					__syncthreads();
				}
				else
				{
					if( tid == 0)
					{
						node = root;
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

__global__ void globalShortstack_TP(struct Rect* _query, int * hit, int* count , int* rootcount, int* pushCount, int* popCount, int mpiSEARCH, int PARTITIONED)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	//int query_id = ( blockIdx.x*blockDim.x) + threadIdx.x;

	int partition_index=0;
	int block_init_position = bid;
	int block_incremental_value = devNUMBLOCK;

	__shared__ struct Node_SOA stack[STACK_SIZE];
	__shared__ struct Rect query;
	__shared__ struct Node_SOA* node;
	__shared__ int numOfoverlapChilds[NODECARD];
	__shared__ int top;
	__shared__ int size;

	__shared__ int t_hit[NODECARD]; 
	__shared__ int childOverlap[NODECARD];

	if( tid == 0)
	{
		top = -1;
		size = 0;
	}
		__syncthreads();

	t_hit[tid] = 0;
	hit[bid] = 0;
	count[bid] = 0;
	pushCount[bid] = 0;
	popCount[bid] = 0;
	rootcount[bid] = 0;

	__syncthreads();

	struct Node_SOA* root = (struct Node_SOA*)deviceRoot[partition_index];

	int passed_hIndex;
	int last_hIndex;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];

		passed_hIndex = 0;
		last_hIndex 	= root->index[root->count-1];


		if( tid == 0)
		{
			node = root;
			rootcount[bid]++;
		}
		__syncthreads();

		while( passed_hIndex < last_hIndex )
		{	
			while( node->level > 0 ) {
				numOfoverlapChilds[tid] = 0;

				if( (tid < node->count) &&
						(node->index[tid] > passed_hIndex) &&
						(dev_Node_SOA_Overlap(&query, node, tid)))
				{
					numOfoverlapChilds[tid] = 1;
//					atomicAdd(&numOfoverlapChilds,1);
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

						numOfoverlapChilds[tid] += numOfoverlapChilds[tid+N];
					}

					N = N/2+N%2;
					__syncthreads();
				}
				if( tid == 0) {
					if(N==1){
						if(childOverlap[0] > childOverlap[1])
							childOverlap[0] = childOverlap[1];

						numOfoverlapChilds[0] += numOfoverlapChilds[1];
					}
				}
				__syncthreads();

				// none of the branches overlapped the query
				if( childOverlap[0] == ( NODECARD+1)) {
					passed_hIndex = node->index[node->count-1];

					//stack.pop()
					if( size > 0 )
					{

						if( tid == 0 )
						{
								node = &stack[top];
								if( top == 0) top = STACK_SIZE-1;
  							else					top--;

								popCount[bid]++;
								
								size--;
						}
						__syncthreads();
					}
					else
					{
						if( tid == 0)
						{
							node = root;
							rootcount[bid]++;
						}
					}
					__syncthreads();
					break;
				} // there exists some overlapped node
				else{

					if( numOfoverlapChilds[0] > 1 )
					{

						//stack.push()
						if( tid == 0 )
						{
							if( top == STACK_SIZE-1) top = 0;
							else										 ++top;

						if( size < STACK_SIZE-1) size++;
						}
						__syncthreads();

						if( node != &stack[top])
						{
							__shared__ char* shared_ptr;
							__shared__ char* global_ptr;

							if( tid == 0)
							{
								pushCount[bid]++;

								shared_ptr = (char*) &stack[top];
								global_ptr = (char*) node;

								memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*12), global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*12), sizeof(int)*2);
							}
							__syncthreads();

							for( int d=0; d<NUMDIMS*2; d++)
								memcpy(shared_ptr+(NODECARD*4)*d+(tid*4), global_ptr+(NODECARD*4)*d+(tid*4), sizeof(float));
							memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+tid*4, global_ptr+(8*NODECARD*NUMDIMS)+tid*4, sizeof(int)); // index
							memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+tid*4, global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+tid*4, sizeof(int));  // half of child pointers
							memcpy(shared_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*8)+tid*4, global_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*8)+tid*4, sizeof(int));  // the other child pointers
							__syncthreads();
						}
					}
					if( tid == 0)
						{
							node = node->child[ childOverlap[0] ];
							count[bid]++;
						}
				}
				__syncthreads();
			}

			if( node->level == 0 )
			{
				if ( tid < node->count
						&& dev_Node_SOA_Overlap(&query, node, tid))
				{
					t_hit[tid]++;
				}
				__syncthreads();

				passed_hIndex = node->index[node->count-1];

				//stack.pop()
				if( size > 0 )
				{
					if( tid == 0 )
					{
						node = &stack[top];

						popCount[bid]++;

						if( top == 0) top = STACK_SIZE-1;
						else					top--;

						size--;
					}
					__syncthreads();
				}
				else
				{
					if( tid == 0)
					{
						node = root;
						rootcount[bid]++;
					}
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

