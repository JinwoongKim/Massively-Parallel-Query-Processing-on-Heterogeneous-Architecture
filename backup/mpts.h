//#####################################################################
//###################### SEARCH FUNCTIONS #############################
//#####################################################################

__device__ bool devRectOverlap(struct Rect *r, struct Rect *s);

void MPTS(int number_of_procs, int myrank);
__global__ void globalMPTS(struct Rect* query, int* hit, int* count, int mpiSEARCH,  int PARTITIONED );
__device__ struct Node* devFindLeftmost(struct Node *root, struct Rect *r, int tid, int* count );
__device__ struct Node* devFindRightmost(struct Node *root, struct Rect *r, int tid, int* count );

__global__ void globalMPTS_ILP(struct Rect* query, int* hit, int* count, int mpiSEARCH,  int PARTITIONED );
__device__ struct Node* devFindLeftmost_ILP(struct Node *root, struct Rect *r, int* count );
__device__ struct Node* devFindRightmost_ILP(struct Node *root, struct Rect *r,int* count );

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

	long t_hit = 0;
	int t_count = 0;

	int* d_hit;
	int* d_count;
	cudaMalloc( (void**) &d_hit,   NUMBLOCK*sizeof(int) );
	cudaMalloc( (void**) &d_count, NUMBLOCK*sizeof(int) );


#ifdef MPI

	struct Rect* d_query;
	cudaMalloc( (void**) &d_query, mpiSEARCH*sizeof(struct Rect) );

	int mpiSEARCH = NUMSEARCH/number_of_procs;
	if( myrank < NUMSEARCH%number_of_procs )
		mpiSEARCH++;

	struct Rect query[mpiSEARCH];

	//work load offset
	fseek(fp, WORKLOAD*NUMSEARCH*sizeof(float)*2*NUMDIMS, SEEK_SET);

	for( int i = 0 ; i < mpiSEARCH; i++){
		//DEBUG
		fseek(fp, myrank*sizeof(float)*2*NUMDIMS + i*number_of_procs*sizeof(float)*2*NUMDIMS, SEEK_SET);
		for(int j=0;j<2*NUMDIMS;j++){
			fread(&query[i].boundary[j], sizeof(float), 1, fp);
		}
	}
	cudaMemcpy(d_query, query, mpiSEARCH*sizeof(struct Rect), cudaMemcpyHostToDevice);
	cudaEventRecord(start_event, 0);

	//MPI_File_read_shared(mpiFileName, &query[nBatch].boundary[0], sizeof(float)*NUMDIMS*2, MPI_FLOAT, &status);
	//MPI_File_read(mpiFileName, &query[nBatch].boundary[0], sizeof(float)*NUMDIMS*2, MPI_FLOAT, &status);

#ifndef ILP
	globalMPTS<<<NUMBLOCK,NODECARD>>>(d_query, d_hit, d_count, mpiSEARCH, PARTITIONED );
#else
	globalMPTS_ILP<<<NUMBLOCK,NUMTHREADS>>>(d_query, d_hit, d_count, mpiSEARCH, PARTITIONED );
#endif

	// Copy hit count from device to host
	cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_count, d_count, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

	for(int j=0;j<NUMBLOCK;j++){
		t_hit += h_hit[j];
		t_count += h_count[j];
	}
#else

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
			globalMPTS<<<nBatch,NODECARD>>>(d_query, d_hit, d_count, nBatch, PARTITIONED );
#else
			globalMPTS_ILP<<<nBatch,NUMTHREADS>>>(d_query, d_hit, d_count, nBatch, PARTITIONED );
#endif

			cudaMemcpy(h_hit, d_hit, nBatch*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, nBatch*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<nBatch;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
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
			globalMPTS<<<NUMBLOCK,NODECARD>>>(d_query, d_hit, d_count, 1, PARTITIONED );
#else
			globalMPTS_ILP<<<NUMBLOCK,NUMTHREADS>>>(d_query, d_hit, d_count, 1, PARTITIONED );
#endif
			cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_count, d_count, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<NUMBLOCK;j++){
				t_hit += h_hit[j];
				t_count += h_count[j];
			}
		}
		cudaThreadSynchronize();
		cudaEventRecord(stop_event, 0) ;
		cudaEventSynchronize(stop_event) ;

		cudaFree( d_query);
	}

#endif

	cudaEventRecord(stop_event, 0) ;
	cudaEventSynchronize(stop_event) ;
	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	printf("MPTS time          : %.3f ms\n", elapsed_time);
	printf("MPTS HIT           : %lu\n",t_hit);
	cout<< "MPTS visited       : " << t_count << endl << endl;


	//selection ratio
#ifdef MPI
	printf("MPTS HIT ratio(%)  : %5.5f\n",((t_hit/mpiSEARCH)/(float)(NUMDATA/100)));
#else
	printf("MPTS HIT ratio(%)  : %5.5f\n",((t_hit/NUMSEARCH)/(float)(NUMDATA/100)));
#endif

	float MIN_time, MAX_time;
	MPI_Reduce(&elapsed_time, &MIN_time, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&elapsed_time, &MAX_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
	if( myrank == 0 ){
		//int MAX_THROUGHPUT, MIN_THROUGHPUT;
		//MAX_THROUGHPUT = ( NUMSEARCH / ( MAX_time/1000));
		//MIN_THROUGHPUT = ( NUMSEARCH / ( MIN_time/1000));
		//printf("MIN %d MAX %d\n", MAX_THROUGHPUT, MIN_THROUGHPUT);
		printf("MAX %.3f MIN %.3f\n", MAX_time, MIN_time);
	}

	fclose(fp);

	cudaFree( d_hit );
	cudaFree( d_count );
}



__global__ void globalMPTS(struct Rect* query, int* hit, int* count, int mpiSEARCH, int PARTITIONED )
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

	hit[bid] = 0;
	count[bid] = 0;
	t_hit[tid]=0; // Initialize the hit count
	__syncthreads();


	for( int n = block_init_position ; n < mpiSEARCH; n += block_incremental_value )
	{

		// Set the range of leaf node which have to find the value
		struct Node* startOff = devFindLeftmost(deviceRoot[partition_index], &query[n], tid, &count[bid]);
		struct Node* endOff = devFindRightmost(deviceRoot[partition_index], &query[n],  tid, &count[bid]);

		/*
			 if ( tid == 0 )
			 if ( startOff > endOff)
			 printf("!!!!!!!!!!!!!!!!!!!!!\n");
			 */


		mbr_ovlp[tid]=0; // Initialize the hit count

		if( startOff == NULL || endOff == NULL) {
			hit[bid] = 0;
			return;
		}


		if(tid==0) {
			count[bid] += (int)((long long)endOff - (long long)startOff)/PGSIZE + 1;
		}
		__syncthreads();
		struct Node* nodePtr = startOff;
		struct Node* leafPtr;

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

			if (tid < nodePtr->count && devRectOverlap(&nodePtr->branch[tid].rect, &query[n])){
				// Hit!
				mbr_ovlp[tid] = 1;
			}
			__syncthreads();

			// based on t_hit[], we should check leaf nodes.

			for(int i=0;i<nodePtr->count;i++){
				if(mbr_ovlp[i] == 1){
					// fetch a leaf node
					if(tid == 0) count[bid] += 1;

					leafPtr = nodePtr->branch[i].child;
					if(tid < leafPtr->count && devRectOverlap(&leafPtr->branch[tid].rect, &query[n])){
						t_hit[tid]++;
					}
				}
			}
			__syncthreads();
			mbr_ovlp[tid] = 0;

			// Set the next node
			nodePtr = (struct Node*) ((char*) nodePtr + PGSIZE);
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

__device__ struct Node* devFindLeftmost(struct Node *root, struct Rect *r, int tid,  int* count )
{

	__shared__ int childOverlap[NODECARD];
	__shared__ struct Node* sn;

	sn = root;

	//if(tid == 0)
	//	printf("sn->level %d \n",sn->level);

	while(sn != NULL) {
		if( tid == 0)  (*count)++;
		if (sn->level > 1) 
		{ // Mark the Overlap thread index
			if (sn->branch[tid].child && devRectOverlap(&sn->branch[tid].rect, r)) {
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
					childOverlap[tid+N] = NODECARD+1;
				}
				N = N/2+N%2;
				__syncthreads();
			}

			// Using the final value
			if( tid == 0) {
				if(N==1){
					if(childOverlap[0] > childOverlap[1]) {
						childOverlap[0] = childOverlap[1];
					}
				}
				// none of the branches overlapped the query
				if( childOverlap[0] == (NODECARD+1)) {
					// option 1. search the rightmost child
					//if( cSIBLING_JUMP == 0){
					sn = sn->branch[ sn->count-1 ].child;
					//}
					// option 2. search the right sibling
					//
					//else if( cSIBLING_JUMP == 1){
					// there's a sibling
					//if( ((struct Node*)((char*) sn + PGSIZE))->level == sn->level )
					//sn = (struct Node*) ((char*) sn + PGSIZE);
					// no sibling
					//else if( ((struct Node*)((char*) sn + PGSIZE))->level != sn->level )
					//	sn = NULL;
					//sn = sn->branch[sn->count-1].child;
					//}
				}
				// there exists some overlapped node
				else{
					sn = sn->branch[ childOverlap[0] ].child;
				}
			}

			__syncthreads();

		}
		else // this is a leaf node 
		{
			return sn;
		}
	}
	return sn;
}

__device__ struct Node* devFindRightmost(struct Node *root, struct Rect *r, int tid,  int* count )
{

	__shared__ int childOverlap[NODECARD];
	__shared__ struct Node* sn;

	sn = root;

	while(sn != NULL){
		if( tid == 0)  (*count)++;
		if (sn->level > 1) // this is an internal node in the tree 
		{
			if (sn->branch[tid].child && devRectOverlap(&sn->branch[tid].rect, r)) {
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
					childOverlap[tid+N] = 0;
				}
				N = N/2+N%2;
				__syncthreads();
			}

			// Using the final value
			if( tid == 0) {
				if(N==1){
					if(childOverlap[0] < childOverlap[1]){
						childOverlap[0] = childOverlap[1];
					}
				}
				// none of the branches overlapped the query
				if( childOverlap[0] == -1 ){
					sn = sn->branch[ 0 ].child;
				}
				// there exists some overlapped node
				else{
					sn = sn->branch[ childOverlap[0] ].child;
				}
			}

			__syncthreads();
		}

		else // this is a leaf node 
		{
			return sn;
		}

	}
	return sn;
}

__global__ void globalMPTS_ILP(struct Rect* query, int* hit, int* count, int mpiSEARCH, int PARTITIONED )
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

	hit[bid] = 0;
	count[bid] = 0;

	for( int thread = tid; thread < NODECARD ; thread+=NUMTHREADS)
		t_hit[thread]=0; // Initialize the hit count

	__syncthreads();


	for( int n = block_init_position ; n < mpiSEARCH; n += block_incremental_value )
	{

		// Set the range of leaf node which have to find the value
		struct Node* startOff = devFindLeftmost_ILP(deviceRoot[partition_index],  &query[n], &count[bid]);
		struct Node* endOff   = devFindRightmost_ILP(deviceRoot[partition_index], &query[n], &count[bid]);

		/*
			 if ( tid == 0 )
			 if ( startOff > endOff)
			 printf("!!!!!!!!!!!!!!!!!!!!!\n");
			 */

		if( startOff == NULL || endOff == NULL) {
			hit[bid] = 0;
			return;
		}


		if( (startOff <=  endOff) &&  tid==0) {
			count[bid] += (int)((long long)endOff - (long long)startOff)/PGSIZE + 1;
		}
		__syncthreads();
		struct Node* nodePtr = startOff;
		struct Node* leafPtr;

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

			for( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS)
			{
				if ( (thread< nodePtr->count ) && ( devRectOverlap(&nodePtr->branch[thread].rect, &query[n]))){
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
					if(tid == 0) count[bid] += 1;

					leafPtr = nodePtr->branch[i].child;

					for( int thread = tid ; thread < leafPtr->count ; thread+=NUMTHREADS)
					{
						if(devRectOverlap(&leafPtr->branch[thread].rect, &query[n])){
							t_hit[thread]++;
						}
					}
				}
			}
			__syncthreads();

			// Set the next node
			nodePtr = (struct Node*) ((char*) nodePtr + PGSIZE);
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

	__syncthreads();

}

__device__ struct Node* devFindLeftmost_ILP(struct Node *root, struct Rect *r, int* count )
{
	int tid = threadIdx.x;

	__shared__ int childOverlap[NODECARD];
	__shared__ struct Node* sn;

	sn = root;

	//if(tid == 0)
	//	printf("sn->level %d \n",sn->level);

	while(sn != NULL) {
		if( tid == 0)  (*count)++;
		if (sn->level > 1) 
		{ // Mark the Overlap thread index
			for( int thread=tid; thread<NODECARD; thread+=NUMTHREADS)
			{
				if (sn->branch[thread].child && devRectOverlap(&sn->branch[thread].rect, r)) {
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
					if(childOverlap[0] > childOverlap[1]) {
						childOverlap[0] = childOverlap[1];
					}
				}
				// none of the branches overlapped the query
				if( childOverlap[0] == (NODECARD+1)) {
					// option 1. search the rightmost child
					//if( cSIBLING_JUMP == 0){
					sn = sn->branch[ sn->count-1 ].child;
					//}
					// option 2. search the right sibling
					//
					//else if( cSIBLING_JUMP == 1){
					// there's a sibling
					//if( ((struct Node*)((char*) sn + PGSIZE))->level == sn->level )
					//sn = (struct Node*) ((char*) sn + PGSIZE);
					// no sibling
					//else if( ((struct Node*)((char*) sn + PGSIZE))->level != sn->level )
					//	sn = NULL;
					//sn = sn->branch[sn->count-1].child;
					//}
				}
				// there exists some overlapped node
				else{
					sn = sn->branch[ childOverlap[0] ].child;
				}
			}

			__syncthreads();

		}
		else // this is a leaf node 
		{
			return sn;
		}
	}
	return sn;
}

__device__ struct Node* devFindRightmost_ILP(struct Node *root, struct Rect *r, int* count )
{
	int tid = threadIdx.x;

	__shared__ int childOverlap[NODECARD];
	__shared__ struct Node* sn;

	sn = root;

	while(sn != NULL){
		if( tid == 0)  (*count)++;
		if (sn->level > 1) // this is an internal node in the tree 
		{
			for( int thread=tid; thread<NODECARD; thread+=NUMTHREADS)
			{
				if (sn->branch[thread].child && devRectOverlap(&sn->branch[thread].rect, r)) {
					childOverlap[thread] = thread;
				}
				else {
					childOverlap[thread] = -1;
				}
			}
			__syncthreads();

			// check if I am the rightmost
			// Gather the Overlap idex and compare
			int N = NODECARD/2+NODECARD%2;
			while(N > 1){
				for( int thread = tid; thread < N ; thread+=NUMTHREADS ) {
					if(childOverlap[thread] < childOverlap[thread+N] )  
						childOverlap[thread] = childOverlap[thread+N];
					childOverlap[thread+N] = 0;
				}
				N = N/2+N%2;
				__syncthreads();
			}

			// Using the final value
			if( tid == 0) {
				if(N==1){
					if(childOverlap[0] < childOverlap[1]){
						childOverlap[0] = childOverlap[1];
					}
				}
				// none of the branches overlapped the query
				if( childOverlap[0] == -1 ){
					sn = sn->branch[ 0 ].child;
				}
				// there exists some overlapped node
				else{
					sn = sn->branch[ childOverlap[0] ].child;
				}
			}

			__syncthreads();
		}

		else // this is a leaf node 
		{
			return sn;
		}

	}
	return sn;
}



