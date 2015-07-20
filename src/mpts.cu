#include <mpts.h>

//#####################################################################
//####################### IMPLEMENTATIONS #############################
//#####################################################################


void MPTS(int number_of_procs, int myrank)
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
	long t_count = 0;
	long t_rootCount = 0;

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

		for( int i = 0 ; i < NUMSEARCH; ){
			int nBatch=0;
			for(nBatch=0; nBatch < NUMBLOCK && i < NUMSEARCH; nBatch++, i++) {
				//DEBUG
				for(int j=0;j<2*NUMDIMS;j++){
					fread(&query[nBatch].boundary[j], sizeof(float), 1, fp);
				}	
			}
			cudaMemcpy(d_query, query, nBatch*sizeof(struct Rect), cudaMemcpyHostToDevice);

#ifndef ILP
			globalMPTS<<<nBatch,NODECARD>>>(d_query, d_hit, d_count, d_rootCount,  nBatch, PARTITIONED );
#else
			globalMPTS_ILP<<<nBatch,NUMTHREADS>>>(d_query, d_hit, d_count, d_rootCount,  nBatch, PARTITIONED );
#endif

			cudaMemcpy(h_hit, d_hit, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, nBatch*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<nBatch;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
				t_rootCount += (long)h_rootCount[j];
			}

		}
		cudaThreadSynchronize();
		cudaEventRecord(stop_event, 0) ;
		cudaEventSynchronize(stop_event) ;

		cudaFree( d_query);
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
			globalMPTS<<<NUMBLOCK,NODECARD>>>(d_query, d_hit, d_count, d_rootCount,  1, PARTITIONED );
#else
			globalMPTS_ILP<<<NUMBLOCK,NUMTHREADS>>>(d_query, d_hit, d_count,  d_rootCount, 1, PARTITIONED );
#endif
			cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rootCount, d_rootCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<NUMBLOCK;j++){
				t_hit += h_hit[j];
				t_count += (long)h_count[j];
				t_rootCount += h_rootCount[j];
			}
		}
		cudaThreadSynchronize();
		cudaEventRecord(stop_event, 0) ;
		cudaEventSynchronize(stop_event) ;

		cudaFree( d_query);
	}


	cudaEventRecord(stop_event, 0) ;
	cudaEventSynchronize(stop_event) ;
	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	printf("MPTS time          : %.3f ms\n", elapsed_time);
	printf("MPTS HIT           : %lu\n",t_hit);
#ifndef ILP
	printf("MPTS visited       : %.3f\n",t_count/(float)(NUMSEARCH));
	printf("MPTS root visited  : %.3f\n", t_rootCount/(float)(NUMSEARCH));
#endif
	printf("MPTS HIT ratio(%)  : %5.5f\n",((t_hit/NUMSEARCH)/(float)(NUMDATA/100)));


	t_time[1] = elapsed_time;
	t_visit[1] = t_count/(float)(NUMSEARCH);
	t_rootVisit[1] = t_rootCount/(float)(NUMSEARCH);
	

	fclose(fp);

	cudaFree( d_hit );
	cudaFree( d_count );
}



__global__ void globalMPTS(struct Rect* _query, int* hit, int* count, int* rootCount, int mpiSEARCH, int PARTITIONED )
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int partition_index=0;
	int block_init_position = bid;
	int block_incremental_value = devNUMBLOCK;

	if( PARTITIONED > 1 ){
		partition_index = bid;
		block_init_position = 0;
		block_incremental_value = 1;
	}

	__shared__ int t_hit[NODECARD]; 
	__shared__ int mbr_ovlp[NODECARD]; 
	__shared__ struct Rect query;

	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;
	t_hit[tid]=0; // Initialize the hit count
	__syncthreads();

	for( int n = block_init_position ; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];


		// Set the range of leaf node which have to find the value
		struct Node_SOA* startOff = (struct Node_SOA*) devFindLeftmost( partition_index, &query, &count[bid]);
		struct Node_SOA* endOff = (struct Node_SOA*) devFindRightmost( partition_index, &query,  &count[bid]);

		if( tid == 0)
			rootCount[bid]+=2;
		__syncthreads();


		mbr_ovlp[tid]=0; // Initialize the hit count

		if( startOff == NULL || endOff == NULL) {
			hit[bid] = 0;
			return;
		}


	//	if(tid==0) {
	//		count[bid] += (int)(((long long)endOff - (long long)startOff)/PGSIZE) + 1;
	//	}
	//	__syncthreads();
		struct Node_SOA* nodePtr = startOff;
		struct Node_SOA* leafPtr;

		//tree height is 1
		//
		//	 if(nodePtr->level == 0){
		//	 if (tid < nodePtr->count && devRectOverlap(&nodePtr->branch[tid].rect, &query[bid]))
		//	 t_hit[tid]=1;
		//
		//			 nodePtr++;
		//			 }


		// Find the hit or not from minimum to maximum
		while(nodePtr<=endOff){
			if( tid == 0) count[bid]++;

			if ((tid < nodePtr->count) &&
					(dev_Node_SOA_Overlap(&query, nodePtr, tid))
					)
			{
				// Hit!
				mbr_ovlp[tid] = 1;
			}
			else
				mbr_ovlp[tid] = 0;

			__syncthreads();

			// based on t_hit[], we should check leaf nodes.

			for(int i=0;i<nodePtr->count;i++){
				if(mbr_ovlp[i] == 1){
					// fetch a leaf node
					if(tid == 0) count[bid]++;

					leafPtr = nodePtr->child[i];
					if( (tid < leafPtr->count ) &&
							(dev_Node_SOA_Overlap(&query, leafPtr, tid)))
					{
						t_hit[tid]++;
					}
				}
			}
			__syncthreads();

			// Set the next node
			nodePtr = (struct Node_SOA*)( (char*) nodePtr+PGSIZE);
			//nodePtr =  nodePtr + 1;
		}
	}
	__syncthreads();

	int N = NODECARD/2 + NODECARD%2;
	while(N > 1){
		if(tid < N ) {
			t_hit[tid] = t_hit[tid] + t_hit[tid+N];
			t_hit[tid+N] = 0;
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

__device__ long devFindLeftmost(int partition_index, struct Rect *r, int* count )
{
	int tid = threadIdx.x;

	struct Node_SOA* sn = (struct Node_SOA*) deviceRoot[partition_index];
	__shared__ int childOverlap[NODECARD];


	while (sn->level > 1) 
	{

		// Mark the Overlap thread index
		if (
				(tid < sn->count) &&
				(dev_Node_SOA_Overlap(r, sn, tid))
				)
		{
			childOverlap[tid] = tid;
		}
		else {
			childOverlap[tid] = NODECARD+1;
		}
		__syncthreads();

		// check if I am the leftmost
		// Gather the Overlap idex and compare
		int N = NODECARD/2 +NODECARD%2;
		while(N > 1){
			if(tid < N ) {
				if(childOverlap[tid] > childOverlap[tid+N] )  
					childOverlap[tid] = childOverlap[tid+N];
			}
			N = N/2+N%2;
			__syncthreads();
		}

		// Using the final value
		if( tid == 0) {
			if(N==1){
				if(childOverlap[0] > childOverlap[1]) 
					childOverlap[0] = childOverlap[1];
			}
		}
		__syncthreads();
		
		// none of the branches overlapped the query
		if( childOverlap[0] == (NODECARD+1))
		{
			// option 1. search the rightmost child
			sn = sn->child[sn->count-1];

		}
		else{ // there exists some overlapped node

			sn = sn->child[ childOverlap[0] ];

		}
		if( tid == 0 ) *(count)++;
		__syncthreads();
	}

	return (long)sn;
}

__device__ long devFindRightmost(int partition_index , struct Rect *r,  int* count )
{
	int tid = threadIdx.x;

	struct Node_SOA* sn = (struct Node_SOA*) deviceRoot[partition_index];
	__shared__ int childOverlap[NODECARD];


		while ( sn->level > 1) // this is an internal node in the tree 
		{

			if (
					(tid < sn->count) &&
					(dev_Node_SOA_Overlap(r, sn, tid))
					)
			{
				childOverlap[tid] = tid;
			}
			else {
				childOverlap[tid] = -1;
			}
			__syncthreads();

			// check if I am the rightmost
			// Gather the Overlap idex and compare
			int N = NODECARD/2+NODECARD%2;
			while(N > 1){
				if(tid < N ) {
					if(childOverlap[tid] < childOverlap[tid+N] )  
						childOverlap[tid] = childOverlap[tid+N];
				}
				N = N/2+N%2;
				__syncthreads();
			}

			// Using the final value
			if( tid == 0) {
				if(N==1){
					if(childOverlap[0] < childOverlap[1])
						childOverlap[0] = childOverlap[1];
				}
			}
			__syncthreads();
			// none of the branches overlapped the query
			if( childOverlap[0] == -1 ){
				sn = sn->child[ 0 ];
			}
			// there exists some overlapped node
			else{
				sn = sn->child[ childOverlap[0] ];
			}

			if( tid == 0 ) *(count)++;
			__syncthreads();

		}

		return (long)sn;
}

__global__ void globalMPTS_ILP(struct Rect* _query, int* hit, int* count, int* rootCount,  int mpiSEARCH, int PARTITIONED )
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
	__shared__ int mbr_ovlp[NODECARD]; 
	__shared__ struct Rect query;

	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	for( int thread = tid; thread < NODECARD ; thread+=NUMTHREADS)
		t_hit[thread]=0; // Initialize the hit count
	__syncthreads();


	for( int n = block_init_position ; n < mpiSEARCH; n += block_incremental_value )
	{
		query = _query[n];

		// Set the range of leaf node which have to find the value
		struct Node_SOA* startOff = (struct Node_SOA*) devFindLeftmost_ILP(partition_index,  &query, &count[bid]);
		struct Node_SOA* endOff   = (struct Node_SOA*) devFindRightmost_ILP(partition_index, &query, &count[bid]);

		if( tid == 0)
			rootCount[bid]+=2;
		__syncthreads();

		/*
			 if ( tid == 0 )
			 if ( startOff > endOff)
			 printf("!!!!!!!!!!!!!!!!!!!!!\n");
			 */

		if( startOff == NULL || endOff == NULL) {
			hit[bid] = 0;
			return;
		}


//		if( (startOff <=  endOff) &&  tid==0) {
//			count[bid] += (int)(((long long)endOff - (long long)startOff)/PGSIZE)+ 1;
//		}
//		__syncthreads();
		struct Node_SOA* nodePtr = startOff;
		struct Node_SOA* leafPtr;

		//tree height is 1
		//
		//	 if(nodePtr->level == 0){
		//	 if (tid < nodePtr->count && devRectOverlap(&nodePtr->branch[tid].rect, &query[bid]))
		//	 t_hit[tid]=1;
		//
		//			 nodePtr++;
		//			 }


		// Find the hit or not from minimum to maximum
		while(nodePtr<=endOff){
			if( tid == 0) count[bid]++;
			for( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS)
			{
				if ( (thread < nodePtr->count) &&
					 ( dev_Node_SOA_Overlap(&query, nodePtr, thread )))	
				{
					// Hit!
					mbr_ovlp[thread] = 1;
				}
				else
				{
					mbr_ovlp[thread] = 0; // Initialize the hit count
				}
			}
			__syncthreads();

			// based on t_hit[], we should check leaf nodes.

			for(int i=0;i<nodePtr->count;i++){
				if(mbr_ovlp[i] == 1){
					// fetch a leaf node
					// !!!
					if(tid == 0) count[bid]++;

					leafPtr = nodePtr->child[i];

					for( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS)
					{
						if( (thread < leafPtr->count) && dev_Node_SOA_Overlap(&query, leafPtr, thread)){
							t_hit[thread]++;
						}
					}
				}
			}
			__syncthreads();

			// Set the next node
			nodePtr++;
		}

	}
	__syncthreads();

	int N = NODECARD/2 + NODECARD%2;
	while(N > 1){
		for( int thread = tid; thread  < N; thread += NUMTHREADS ){
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

__device__ long devFindLeftmost_ILP(int partition_index , struct Rect *r, int* count )
{
	int tid = threadIdx.x;

	struct Node_SOA* sn = (struct Node_SOA*) deviceRoot[partition_index];
	__shared__ int childOverlap[NODECARD];


	while (sn->level > 1) 
	{

		// Mark the Overlap thread index
		for( int thread=tid; thread<NODECARD; thread+=NUMTHREADS)
		{
		if (
				(thread < sn->count) &&
				(dev_Node_SOA_Overlap(r, sn, thread))
				)
		{
			childOverlap[thread] = thread;
		}
		else {
			childOverlap[thread] = NODECARD+1;
		}
		}
		__syncthreads();

		// check if I am the leftmost
		// Gather the Overlap idex and compare
		int N = NODECARD/2 +NODECARD%2;
		while(N > 1){
			for( int thread = tid; thread < N ; thread+=NUMTHREADS ) {
				if(childOverlap[thread] > childOverlap[thread+N] )  
					childOverlap[thread] = childOverlap[thread+N];
				childOverlap[thread+N] = NODECARD+1;
			}
			N = N/2+N%2;
			__syncthreads();
		}

		// Using the final value
		if( tid == 0) {
			if(N==1){
				if(childOverlap[0] > childOverlap[1]) 
					childOverlap[0] = childOverlap[1];
			}
		}
		__syncthreads();
		
		// none of the branches overlapped the query
		if( childOverlap[0] == (NODECARD+1))
		{
			// option 1. search the rightmost child
			sn = sn->child[sn->count-1];

		}
		else{ // there exists some overlapped node

			sn = sn->child[ childOverlap[0] ];

		}
		if( tid == 0 ) *(count)++;
		__syncthreads();
	}

	return (long)sn;
}

__device__ long devFindRightmost_ILP(int partition_index ,struct Rect *r, int* count )
{
	int tid = threadIdx.x;

	struct Node_SOA* sn = (struct Node_SOA*) deviceRoot[partition_index];
	__shared__ int childOverlap[NODECARD];


	while (sn->level > 1) 
	{

		// Mark the Overlap thread index
		for( int thread=tid; thread<NODECARD; thread+=NUMTHREADS)
		{
			if (
					(thread < sn->count) &&
					(dev_Node_SOA_Overlap(r, sn, thread))
				 )
			{
				childOverlap[thread] = thread;
			}
			else {
				childOverlap[thread] = NODECARD+1;
			}
		}
		__syncthreads();

		// check if I am the leftmost
		// Gather the Overlap idex and compare
		int N = NODECARD/2 +NODECARD%2;
		while(N > 1){
			for( int thread = tid; thread < N ; thread+=NUMTHREADS ) {
				if(childOverlap[thread] > childOverlap[thread+N] )  
					childOverlap[thread] = childOverlap[thread+N];
				childOverlap[thread+N] = NODECARD+1;
			}
			N = N/2+N%2;
			__syncthreads();
		}

		// Using the final value
		if( tid == 0) {
			if(N==1){
				if(childOverlap[0] > childOverlap[1]) 
					childOverlap[0] = childOverlap[1];
			}
		}
		__syncthreads();
		
		// none of the branches overlapped the query
		if( childOverlap[0] == -1)
		{
			// option 1. search the rightmost child
			sn = sn->child[0];

		}
		else{ // there exists some overlapped node

			sn = sn->child[ childOverlap[0] ];

		}
		if( tid == 0 ) *(count)++;
		__syncthreads();
	}

	return (long)sn;

} 


