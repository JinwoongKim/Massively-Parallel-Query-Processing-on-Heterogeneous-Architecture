#include <parentLink.h>

//#####################################################################
//####################### IMPLEMENTATIONS #############################
//#####################################################################



void ParentLink(int number_of_procs, int myrank)
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
	int h_parentCount[NUMBLOCK];

	long t_hit = 0;
	int t_count = 0;
	int t_rootCount = 0;
	int t_parentCount = 0;

	int* d_hit;
	int* d_count;
	int* d_rootCount;
	int* d_parentCount;

	cudaMalloc( (void**) &d_hit,   NUMBLOCK*sizeof(int) );
	cudaMalloc( (void**) &d_count, NUMBLOCK*sizeof(int) );
	cudaMalloc( (void**) &d_rootCount, NUMBLOCK*sizeof(int) );
	cudaMalloc( (void**) &d_parentCount, NUMBLOCK*sizeof(int) );


//	globalDesignTraversalScenarioBVH<<<1, NODECARD>>>();


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
			globalParentLink_ILP_BVH<<<nBatch, NUMTHREADS>>>(d_query, d_hit, nBatch, PARTITIONED );
#else
#ifdef TP
			globalParentLink_BVH_TP<<<NUMBLOCK, NUM_TP>>>(d_query, d_hit, d_count, d_rootCount, d_parentCount, nBatch, PARTITIONED );
#else
			globalParentLink_BVH<<<nBatch, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, d_parentCount, nBatch, PARTITIONED );
#endif
#endif


#ifdef TP
			cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_parentCount, d_parentCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<NUMBLOCK;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
				t_rootCount += h_rootCount[j]; 
				t_parentCount += h_parentCount[j]; 
			}
#else
			cudaMemcpy(h_hit, d_hit, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_parentCount, d_parentCount, nBatch*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<nBatch;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
				t_rootCount += h_rootCount[j]; 
				t_parentCount += h_parentCount[j]; 
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
				//globalParentLink_BVH<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, d_fromChildCount, d_fromSiblingCount, d_fromParentCount, 1, PARTITIONED );
				globalParentLink_BVH<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_rootCount,d_parentCount, 1, PARTITIONED );
#else
				globalParentLink_ILP_BVH<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, 1, PARTITIONED );
#endif

			cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_parentCount, d_parentCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<NUMBLOCK;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
				t_rootCount += h_rootCount[j]; 
				t_parentCount += h_parentCount[j]; 
			}
		}
		cudaThreadSynchronize();
		cudaEventRecord(stop_event, 0) ;
		cudaEventSynchronize(stop_event) ;

		cudaFree( d_query );
	}


	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	printf("ParentLink time          : %.3f ms\n", elapsed_time);
	printf("ParentLink HIT           : %lu\n",t_hit);
#ifndef ILP
	printf("ParentLink visited       : %.6f\n" , (t_count)/(float)NUMSEARCH );
	printf("ParentLink root visited  : %.6f\n" , (t_rootCount)/(float)NUMSEARCH );
	printf("ParentLink parnetCount   : %.6f\n" , (t_parentCount)/(float)NUMSEARCH);
#endif
	printf("ParentLink HIT ratio(%)  : %5.5f\n",((t_hit/NUMSEARCH)/(float)(NUMDATA/100)));

	t_time[5] = elapsed_time;
	t_visit[5] = (t_count)/(float)NUMSEARCH;
	t_rootVisit[5] = (t_rootCount)/(float)NUMSEARCH;
	t_parent[5] = (t_parentCount)/(float)NUMSEARCH;

	fclose(fp);

	cudaFree( d_hit );
	cudaFree( d_count );
	cudaFree( d_rootCount );
	cudaFree( d_parentCount );

}




__global__ void globalParentLink_BVH(struct Rect* _query, int * hit, int* count , int* rootCount, int* parentCount, int mpiSEARCH, int PARTITIONED  )
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
	__shared__  struct Rect query;
	BVH_Node_SOA* node_ptr;

	t_hit[tid] = 0;
	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;
	parentCount[bid] = 0;

	BVH_Node_SOA* root = (BVH_Node_SOA*) deviceBVHRoot[partition_index];


	unsigned long long passed_mortonCode;
	unsigned long long last_mortonCode;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_mortonCode = 0;
		last_mortonCode 	= root->index[root->count-1];

		node_ptr 			= root;
		if( tid == 0 )
			rootCount[bid]++;
		__syncthreads();

		while( passed_mortonCode < last_mortonCode )
		{	
			//find out left most child node till leaf level
			if( node_ptr->level > 0 ) {
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

					if(node_ptr != root)
						node_ptr = node_ptr->parent;

					if( tid == 0 )
					{
						if( node_ptr == root)
							rootCount[bid]++;
						else
							parentCount[bid]++;
					}
					__syncthreads();
				}
				// there exists some overlapped node
				else{
					node_ptr = node_ptr->child[ childOverlap[0] ];
					count[bid]++;
				}
			}

			if( node_ptr->level == 0 )
			{
				if ( tid < node_ptr->count && dev_BVH_Node_SOA_Overlap(&query, node_ptr, tid))
				{
					t_hit[tid]++;
				}
				__syncthreads();

				passed_mortonCode = node_ptr->index[node_ptr->count-1];

				//last leaf node

				if ( node_ptr->index[node_ptr->count-1]== last_mortonCode )
					break;
				else
				{
					node_ptr = node_ptr->parent;

					if( tid == 0 )
					{
						if( node_ptr == root)
							rootCount[bid]++;
						else
							parentCount[bid]++;
					}
					__syncthreads();
				}
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


/* parent link with state machine
__global__ void globalParentLink_BVH(struct Rect* _query, int * hit, int* count , int* rootCount, int* fromChildCount, int* fromSiblingCount, int* fromParentCount, int mpiSEARCH, int PARTITIONED  )
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
	//__shared__ bool isHit;

	t_hit[tid] = 0;
	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;
	fromChildCount[bid] = 0;
	fromSiblingCount[bid] = 0;
	fromParentCount[bid] = 0;

	BVH_Node* root = deviceBVHRoot[partition_index];

  unsigned long long passed_mortonCode;
  unsigned long long last_mortonCode;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_mortonCode = 0;
		last_mortonCode 	= root->branch[root->count-1].mortonCode;

		BVH_Node* node_ptr 			= root;

		// 0 : fromChild
		// 1 : fromSibling
		// 2 : fromParent
		
		int state = 2; 
		if( tid == 0 ) fromParentCount[bid]++;

		while( passed_mortonCode < last_mortonCode)
		{

			if( tid == 0)
			{
				if ( node_ptr == root) rootCount[bid]++;
				else                   count[bid]++;
			}

			switch(state)
			{
				//fromChild -> sibling or parent
				case 0:
					// go to sibling
					if( (node_ptr+1)->parent == node_ptr->parent)
					{
						node_ptr++;
						state = 1; //fromSibling
						if( tid == 0 ) fromSiblingCount[bid]++;
					}
					// go to parent
					else if(node_ptr->parent != 0x0)
					{
						node_ptr = node_ptr->parent;
						state = 0; //fromChild
						if( tid == 0 ) fromChildCount[bid]++;
					}
					break;
				case 1: //fromSibling -> child or parent or sibling
				case 2: //fromParent -> child or parent or sibling
					//Internal node
					if(  node_ptr->level > 0)
					{
						if( (tid < node_ptr->count) &&
								(node_ptr->branch[tid].mortonCode > passed_mortonCode) &&
								(devRectOverlap(&node_ptr->branch[tid].rect, &query)))
						{
							childOverlap[tid] = tid;
						}else{
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

						//No overlap child -> sibling or parent
						if( childOverlap[0] == ( NODECARD+1))
						{
							passed_mortonCode = node_ptr->branch[node_ptr->count-1].mortonCode;
							if( passed_mortonCode == last_mortonCode) break;
						}
						else //overlap child
						{
							node_ptr = node_ptr->branch[ childOverlap[0] ].child;
							state = 2; //fromParent
							if( tid == 0 ) fromParentCount[bid]++;
							break;
						}
					}
					else// if( node_ptr->level == 0  )
					{
						if ( (tid < node_ptr->count) && (devRectOverlap(&node_ptr->branch[tid].rect, &query)))
						{
							t_hit[tid]++;
						}
						__syncthreads();

						passed_mortonCode = node_ptr->branch[node_ptr->count-1].mortonCode;
						if( passed_mortonCode == last_mortonCode) break;
					} 

					//If there sibling node exist, go to sibling node
					if( (node_ptr+1)->parent == node_ptr->parent)
					{
						node_ptr++;
						state = 1; //fromSibling
						if( tid == 0 ) fromSiblingCount[bid]++;
					} // or go to parent
					else if( node_ptr->parent != 0x0)
					{
						node_ptr = node_ptr->parent;
						state = 0; //fromChild
						if( tid == 0 ) fromChildCount[bid]++;
					}
					break;
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
*/
__global__ void globalParentLink_ILP_BVH(struct Rect* _query, int * hit, int mpiSEARCH, int PARTITIONED)
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

	__shared__ int t_hit[NUMTHREADS]; 
	__shared__ int childOverlap[NUMTHREADS];
	__shared__  struct Rect query;
	BVH_Node_SOA* node_ptr;


	t_hit[tid] = 0;

	hit[bid] = 0;

	BVH_Node_SOA* root = (BVH_Node_SOA*) deviceBVHRoot[partition_index];

	int passed_Index;
	int last_Index;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_Index = 0;
		last_Index 	= root->index[root->count-1];

		node_ptr 			= root;

		while( passed_Index < last_Index )
		{	
			//find out left most child node till leaf level
			if( node_ptr->level > 0 ) {

				__shared__ bool isOverlap;
				isOverlap = false;

				for ( int thread = tid ; thread < NODECARD ; thread+=NUMTHREADS){
					if( (thread < node_ptr->count) &&
							(node_ptr->index[thread]> passed_Index) &&
							(dev_BVH_Node_SOA_Overlap(&query, node_ptr, thread ))){
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
					passed_Index = node_ptr->index[node_ptr->count-1];

					if( node_ptr != root)
						node_ptr = node_ptr->parent;

				}
				// there exists some overlapped node
				else{
					node_ptr = node_ptr->child[ childOverlap[0] ];
				}
			__syncthreads();
			}

			if( node_ptr->level == 0 )
			{

				for ( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS){
					if ( thread < node_ptr->count && 
							dev_BVH_Node_SOA_Overlap(&query, node_ptr, thread))
					{
						t_hit[tid]++;
					}
				}
				__syncthreads();

				passed_Index = node_ptr->index[node_ptr->count-1];

				//last leaf node

				if ( node_ptr->index[node_ptr->count-1]== last_Index )
					break;
				else
				{
					node_ptr = node_ptr->parent;
				}
			}
		}
	}
	int N = NUMTHREADS/2 + NUMTHREADS%2;
	while(N > 1){
		if( tid < N ){
				t_hit[tid] = t_hit[tid+N];
		}
		N = N/2+N%2;
		__syncthreads();
	}

	if(tid==0) {
		if(N==1) 
			hit[bid] = t_hit[0] + t_hit[1];
		else
			hit[bid] = t_hit[0];
	}
}
/*
__global__ void globalParentLink_ILP_BVH(struct Rect* _query, int * hit, int mpiSEARCH, int PARTITIONED)
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

	BVH_Node* root = deviceBVHRoot[partition_index];

  unsigned long long passed_mortonCode;
  unsigned long long last_mortonCode;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_mortonCode = 0;
		last_mortonCode 	= root->branch[root->count-1].mortonCode;

		BVH_Node* node_ptr 			= root;

		// 0 : fromChild
		// 1 : fromSibling
		// 2 : fromParent
		
		int state = 2; 

		while( passed_mortonCode < last_mortonCode)
		{
			switch(state)
			{
				//fromChild -> sibling or parent
				case 0:

					if( (node_ptr+1)->parent == node_ptr->parent)
					{
						node_ptr++;
						state = 1; //fromSibling
					}
					else if(node_ptr->parent != 0x0)
					{
						node_ptr = node_ptr->parent;
						state = 0; //fromChild
					}
					break;
				case 1: //fromSibling
				case 2: //fromParent
					//Internal node
					if(  node_ptr->level > 0)
					{
						for ( int thread = tid ; thread < NODECARD ; thread+=NUMTHREADS){
							if( (thread < node_ptr->count) &&
									(node_ptr->branch[thread].mortonCode > passed_mortonCode) &&
									(devRectOverlap(&node_ptr->branch[thread].rect, &query)))
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
							passed_mortonCode = node_ptr->branch[node_ptr->count-1].mortonCode;
							if( passed_mortonCode == last_mortonCode) break;
						}
						else //overlap child
						{
							node_ptr = node_ptr->branch[ childOverlap[0] ].child;
							state = 2; //fromParent
							break;
						}
					}
					else// if( node_ptr->level == 0  )
					{

						for ( int thread = tid ; thread < node_ptr->count ; thread+=NUMTHREADS){
							if ( devRectOverlap(&node_ptr->branch[thread].rect, &query))
							{
								t_hit[thread]++;
							}
						}
						__syncthreads();

						passed_mortonCode = node_ptr->branch[node_ptr->count-1].mortonCode;
						if( passed_mortonCode == last_mortonCode) break;
					} 
					//Whatever node is internal or leaf,
					if( (node_ptr+1)->parent == node_ptr->parent)
					{
						node_ptr++;
						state = 1; //fromSibling
					}
					else if( node_ptr->parent != 0x0)
					{
						node_ptr = node_ptr->parent;
						state = 0; //fromChild
					}
					break;
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
*/
__global__ void globalParentLink_BVH_TP(struct Rect* _query, int * hit, int* count , int* rootCount, int* parentCount, int mpiSEARCH, int PARTITIONED  )
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
	__shared__ int t_parentCount[NUM_TP]; 

	struct Rect query;
	BVH_Node_SOA* node_ptr;

	t_hit[tid] = 0;
	t_rootCount[tid] = 0;
	t_count[tid] = 0;
	t_parentCount[tid] = 0;

	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;
	parentCount[bid] = 0;

	BVH_Node_SOA* root = (BVH_Node_SOA*) deviceBVHRoot[partition_index];


	int passed_Index;
	int last_Index;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];
		passed_Index = 0;
		last_Index 	= root->index[root->count-1];

		node_ptr 			= root;
		t_rootCount[tid]++;

		while( passed_Index < last_Index )
		{	
			//find out left most child node till leaf level
			if( node_ptr->level > 0 ) {

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
				if( t == node_ptr->count ) {
					passed_Index = node_ptr->index[node_ptr->count-1];

					if(node_ptr != root)
						node_ptr = node_ptr->parent;

					if( node_ptr == root)
						t_rootCount[tid]++;
					else
						t_parentCount[tid]++;
				}
				// there exists some overlapped node
				else{
					node_ptr = node_ptr->child[ t ];
					t_count[tid]++;
				}
			}

			if( node_ptr->level == 0 )
			{

				int t;
				for( t = 0; t < node_ptr->count; t++)
				{
					if ( dev_BVH_Node_SOA_Overlap(&query, node_ptr, t))
					{
						t_hit[tid]++;
					}
				}

				passed_Index = node_ptr->index[node_ptr->count-1];

				//last leaf node

				if ( node_ptr->index[node_ptr->count-1]== last_Index )
					break;
				else
				{
					node_ptr = node_ptr->parent;

					if( node_ptr == root)
						t_rootCount[tid]++;
					else
						t_parentCount[tid]++;
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
			t_parentCount[tid] = t_parentCount[tid] + t_parentCount[tid+N];
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
			parentCount[bid] = t_parentCount[0] + t_parentCount[1];
		}
		else
		{
			hit[bid] = t_hit[0];
			rootCount[bid] = t_rootCount[0];
			count[bid] = t_count[0];
			parentCount[bid] = t_parentCount[0];
		}
	}
}
