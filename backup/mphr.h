//#####################################################################
//###################### SEARCH FUNCTIONS #############################
//#####################################################################


__device__ bool devRectOverlap(struct Rect *r, struct Rect *s);

void MPHR(int number_of_procs, int myrank);
__global__ void globalMPHR(struct Rect* query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED  );
__global__ void globalMPHR_ILP(struct Rect* query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED  );

void MPHR2(int number_of_procs, int myrank);
__global__ void globalMPHR2(struct Rect* query, int * hit, int* count , long* _leafNode_offset, long* _extendLeafNode_offset , int* rootCount, int mpiSEARCH, int boundary_of_trees, int PARTITIONED  );
__global__ void globalMPHR2_ILP(struct Rect* query, int * hit, int* count , long* _leafNode_offset, long* _extendLeafNode_offset , int* rootCount, int mpiSEARCH, int boundary_of_trees, int PARTITIONED  );

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

#ifdef MPI

	//assign quries to each gpu device
	int mpiSEARCH = NUMSEARCH/number_of_procs;
	if( myrank < NUMSEARCH%number_of_procs )
		mpiSEARCH++;


	struct Rect* d_query;
	cudaMalloc( (void**) &d_query, mpiSEARCH*sizeof(struct Rect) );

	struct Rect query[mpiSEARCH];

	//work load offset
	fseek(fp, WORKLOAD*NUMSEARCH*sizeof(float)*2*NUMDIMS, SEEK_SET);

	for( int i = 0 ; i < mpiSEARCH; i++){
		fseek(fp, myrank*sizeof(float)*2*NUMDIMS + i*number_of_procs*sizeof(float)*2*NUMDIMS, SEEK_SET);
		for(int j=0;j<2*NUMDIMS;j++){
			fread(&query[i].boundary[j], sizeof(float), 1, fp);
		}
	}
	cudaMemcpy(d_query, query, mpiSEARCH*sizeof(struct Rect), cudaMemcpyHostToDevice);
	cudaEventRecord(start_event, 0);
#ifndef ILP
	globalMPHR<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, mpiSEARCH, PARTITIONED );
#else
	globalMPHR_ILP<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, d_count, d_rootCount, mpiSEARCH, PARTITIONED );
#endif

	// Copy hit count from device to host
	cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_count, d_count, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rootCount, d_rootCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

	for(int j=0;j<NUMBLOCK;j++){
		t_hit += h_hit[j];
		t_count += h_count[j];
		t_rootCount += h_rootCount[j]; 
	}

	cudaEventRecord(stop_event, 0) ;
	cudaEventSynchronize(stop_event) ;

	cudaFree( d_query );
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
			globalMPHR<<<nBatch, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, nBatch, PARTITIONED );
#else
			globalMPHR_ILP<<<nBatch, NUMTHREADS>>>(d_query, d_hit, d_count, d_rootCount, nBatch, PARTITIONED );
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
			globalMPHR<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_rootCount, 1, PARTITIONED );
#else
			globalMPHR_ILP<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, d_count, d_rootCount, 1, PARTITIONED );
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
#endif


	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	printf("MPHR time          : %.3f ms\n", elapsed_time);
	printf("MPHR HIT           : %lu\n",t_hit);
#ifdef MPI
	cout<< "MPHR visited       : " << t_count-mpiSEARCH << endl;
	cout<< "MPHR root visited  : " << t_rootCount+mpiSEARCH << endl;
#else
	cout<< "MPHR visited       : " << t_count-NUMSEARCH << endl;
	cout<< "MPHR root visited  : " << t_rootCount+NUMSEARCH << endl;
#endif

	//selection ratio
#ifdef MPI
	printf("MPHR HIT ratio(%)  : %5.5f\n",((t_hit/mpiSEARCH)/(float)(NUMDATA/100)));
#else
	printf("MPHR HIT ratio(%)  : %5.5f\n",((t_hit/NUMSEARCH)/(float)(NUMDATA/100)));
#endif


	float MIN_time, MAX_time;
	MPI_Reduce(&elapsed_time, &MIN_time, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&elapsed_time, &MAX_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
	if( myrank == 0 )
		printf("MAX %.3f MIN %.3f\n", MAX_time, MIN_time);

	fclose(fp);

	cudaFree( d_hit );
	cudaFree( d_count );
	cudaFree( d_rootCount );

}

__global__ void globalMPHR (struct Rect* query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED)
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

	t_hit[tid] = 0;
	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	struct Node* root = deviceRoot[partition_index];

	bitmask_t passed_hIndex;
	bitmask_t last_hIndex;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		passed_hIndex = 0;
		last_hIndex 	= root->branch[root->count-1].hIndex;

		struct Node* node_ptr 			= root;

		while( passed_hIndex < last_hIndex )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

				if ( tid == 0) count[bid]++;

				if( (tid < node_ptr->count) &&
						(node_ptr->branch[tid].hIndex > passed_hIndex) &&
						(devRectOverlap(&node_ptr->branch[tid].rect, &query[n])))
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
					passed_hIndex = node_ptr->branch[node_ptr->count-1].hIndex;

					if( tid == 0 ) rootCount[bid]++;
					node_ptr = root;
					break;
				}
				// there exists some overlapped node
				else{
					node_ptr = node_ptr->branch[ childOverlap[0] ].child;
				}
				__syncthreads();
			}


			while( node_ptr->level == 0 )
			{
				if ( tid == 0) count[bid]++;

				isHit = false;

				if ( tid < node_ptr->count && devRectOverlap(&node_ptr->branch[tid].rect, &query[n]))
				{
					t_hit[tid]++;
					isHit = true;
				}
				__syncthreads();

				passed_hIndex = node_ptr->branch[node_ptr->count-1].hIndex;

				//last leaf node

				if ( node_ptr->branch[node_ptr->count-1].hIndex == last_hIndex )
					break;
				else if( isHit )
				{
					node_ptr++;
				}
				else
				{
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
__global__ void globalMPHR_ILP(struct Rect* query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED)
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

	for ( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS)
		t_hit[thread] = 0;

	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	struct Node* root = deviceRoot[partition_index];

	bitmask_t passed_hIndex;
	bitmask_t last_hIndex;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		passed_hIndex = 0;
		last_hIndex 	= root->branch[root->count-1].hIndex;

		struct Node* node_ptr 			= root;

		while( passed_hIndex < last_hIndex )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

				if ( tid == 0) count[bid]++;

				for ( int thread = tid ; thread < NODECARD ; thread+=NUMTHREADS){
					if( (thread < node_ptr->count) &&
							(node_ptr->branch[thread].hIndex > passed_hIndex) &&
							(devRectOverlap(&node_ptr->branch[thread].rect, &query[n]))){
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
					passed_hIndex = node_ptr->branch[node_ptr->count-1].hIndex;

					if( tid == 0 ) rootCount[bid]++;
					node_ptr = root;
					break;
				}
				// there exists some overlapped node
				else{
					node_ptr = node_ptr->branch[ childOverlap[0] ].child;
				}
				__syncthreads();
			}



			while( node_ptr->level == 0 )
			{
				if ( tid == 0) count[bid]++;

				isHit = false;


				for ( int thread = tid ; thread < node_ptr->count ; thread+=NUMTHREADS){
					if ( devRectOverlap(&node_ptr->branch[thread].rect, &query[n]))
					{
						t_hit[thread]++;
						isHit = true;
					}
				}
				__syncthreads();

				passed_hIndex = node_ptr->branch[node_ptr->count-1].hIndex;

				//last leaf node

				if ( node_ptr->branch[node_ptr->count-1].hIndex == last_hIndex )
					break;
				else if( isHit )
				{
					node_ptr++;
				}
				else
				{
					if( tid == 0 ) rootCount[bid]++;
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

	long leafNode_offset[2];
	leafNode_offset[0] = (indexSize[0]/PGSIZE) - number_of_node_in_level[0][0];
	leafNode_offset[1] = (indexSize[1]/PGSIZE) - number_of_node_in_level[1][0];

	long extendLeafNode_offset[2];
	extendLeafNode_offset[0] = leafNode_offset[0] - number_of_node_in_level[0][1];
	extendLeafNode_offset[1] = leafNode_offset[1] - number_of_node_in_level[1][1];


	long* d_leafNode_offset;
	cudaMalloc((void**)&d_leafNode_offset, sizeof(long)*2);
	long* d_extendLeafNode_offset;
	cudaMalloc((void**)&d_extendLeafNode_offset, sizeof(long)*2);

	cudaMemcpy( d_leafNode_offset, leafNode_offset, sizeof(long)*2, cudaMemcpyHostToDevice);
	cudaMemcpy( d_extendLeafNode_offset, extendLeafNode_offset, sizeof(long)*2, cudaMemcpyHostToDevice);


#ifdef MPI

	//assign quries to each gpu device
	int mpiSEARCH = NUMSEARCH/number_of_procs;
	if( myrank < NUMSEARCH%number_of_procs )
		mpiSEARCH++;

	struct Rect* d_query;
	cudaMalloc( (void**) &d_query, mpiSEARCH*sizeof(struct Rect) );


	struct Rect query[mpiSEARCH];

	//work load offset
	fseek(fp, WORKLOAD*NUMSEARCH*sizeof(float)*2*NUMDIMS, SEEK_SET);

	for( int i = 0 ; i < mpiSEARCH; i++){
		fseek(fp, myrank*sizeof(float)*2*NUMDIMS + i*number_of_procs*sizeof(float)*2*NUMDIMS, SEEK_SET);
		for(int j=0;j<2*NUMDIMS;j++){
			fread(&query[i].boundary[j], sizeof(float), 1, fp);
		}
	}
	cudaMemcpy(d_query, query, mpiSEARCH*sizeof(struct Rect), cudaMemcpyHostToDevice);
	cudaEventRecord(start_event, 0);

#ifndef ILP
	globalMPHR2<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_leafNode_offset, d_extendLeafNode_offset , d_rootCount, mpiSEARCH, boundary_of_trees, PARTITIONED );
#else
	globalMPHR2_ILP<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, d_count, d_leafNode_offset, d_extendLeafNode_offset , d_rootCount, mpiSEARCH, boundary_of_trees, PARTITIONED );
#endif

	// Copy hit count from device to host
	cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_count, d_count, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rootCount, d_rootCount, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);

	for(int j=0;j<NUMBLOCK;j++){
		t_hit += h_hit[j];
		t_count += h_count[j];
		t_rootCount += h_rootCount[j]; 
	}
	cudaEventRecord(stop_event, 0) ;
	cudaEventSynchronize(stop_event) ;
	cudaFree( d_query );

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
			globalMPHR2<<<nBatch, NODECARD>>>(d_query, d_hit, d_count, d_leafNode_offset, d_extendLeafNode_offset , d_rootCount, nBatch, boundary_of_trees, PARTITIONED );
#else
			globalMPHR2_ILP<<<nBatch, NUMTHREADS>>>(d_query, d_hit, d_count, d_leafNode_offset, d_extendLeafNode_offset , d_rootCount, nBatch, boundary_of_trees, PARTITIONED );
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
			globalMPHR2<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_count, d_leafNode_offset, d_extendLeafNode_offset , d_rootCount, 1, boundary_of_trees, PARTITIONED );
#else
			globalMPHR2_ILP<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, d_count, d_leafNode_offset, d_extendLeafNode_offset , d_rootCount, 1, boundary_of_trees, PARTITIONED );
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

#endif

	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	printf("MPHR2 time          : %.3f ms\n", elapsed_time);
	printf("MPHR2 HIT           : %lu\n",t_hit);
#ifdef MPI
	cout<< "MPHR2 visited       : " << t_count-mpiSEARCH << endl;
	cout<< "MPHR2 root visited  : " << t_rootCount+mpiSEARCH << endl;
#else
	cout<< "MPHR2 visited       : " << t_count-NUMSEARCH << endl;
	cout<< "MPHR2 root visited  : " << t_rootCount+NUMSEARCH << endl;
#endif

	//selection ratio
#ifdef MPI
	printf("MPHR2 HIT ratio(%)  : %5.5f\n",((t_hit/mpiSEARCH)/(float)(NUMDATA/100)));
#else
	printf("MPHR2 HIT ratio(%)  : %5.5f\n",((t_hit/NUMSEARCH)/(float)(NUMDATA/100)));
#endif


	float MIN_time, MAX_time;
	MPI_Reduce(&elapsed_time, &MIN_time, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&elapsed_time, &MAX_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
	if( myrank == 0 )
		printf("MAX %.3f MIN %.3f\n", MAX_time, MIN_time);

	fclose(fp);

	cudaFree( d_hit );
	cudaFree( d_count );
	cudaFree( d_rootCount );


}




__global__ void globalMPHR2(struct Rect* query, int * hit, int* count , long* _leafNode_offset, long* _extendLeafNode_offset , int* rootCount, int mpiSEARCH, int boundary_index, int PARTITIONED)
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

	t_hit[tid] = 0;
	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	struct Node* root = deviceRoot[partition_index];

	struct Node* leafNode_ptr   = (struct Node*) ( (char*) root+(PGSIZE*leafNode_offset) );
	struct Node* extendNode_ptr = (struct Node*) ( (char*) root+(PGSIZE*extendLeafNode_offset) );

	bitmask_t passed_hIndex;
	bitmask_t last_hIndex;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		passed_hIndex = 0;
		last_hIndex 	= root->branch[root->count-1].hIndex;

		struct Node* node_ptr 			= root;

		while( passed_hIndex < last_hIndex )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

				if ( tid == 0) count[bid]++;

				if( (tid < node_ptr->count) &&
						(node_ptr->branch[tid].hIndex > passed_hIndex) &&
						(devRectOverlap(&node_ptr->branch[tid].rect, &query[n])))
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
					passed_hIndex = node_ptr->branch[node_ptr->count-1].hIndex;

					if( tid == 0 ) rootCount[bid]++;
					node_ptr = root;
					break;
				}
				// there exists some overlapped node
				else{
					node_ptr = node_ptr->branch[ childOverlap[0] ].child;
				}
				__syncthreads();
			}


			while( node_ptr->level == 0 )
			{
				if ( tid == 0) count[bid]++;

				isHit = false;


				if ( tid < node_ptr->count && devRectOverlap(&node_ptr->branch[tid].rect, &query[n]))
				{
					t_hit[tid]++;
					isHit = true;
				}
				__syncthreads();

				passed_hIndex = node_ptr->branch[node_ptr->count-1].hIndex;

				//last leaf node

				if ( node_ptr->branch[node_ptr->count-1].hIndex == last_hIndex )
					break;
				else if( isHit )
				{
					node_ptr++;
				}
				else
				{
					node_ptr = extendNode_ptr + ( ( node_ptr - leafNode_ptr) / NODECARD) ;
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


__global__ void globalMPHR2_ILP(struct Rect* query, int * hit, int* count , long* _leafNode_offset, long* _extendLeafNode_offset , int* rootCount, int mpiSEARCH, int boundary_index, int PARTITIONED)
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

	for ( int thread = tid ; thread < NODECARD; thread+=NUMTHREADS )
		t_hit[thread] = 0;

	hit[bid] = 0;
	count[bid] = 0;
	rootCount[bid] = 0;

	struct Node* root = deviceRoot[partition_index];
	struct Node* leafNode_ptr   = (struct Node*) ( (char*) root+(PGSIZE*leafNode_offset) );
	struct Node* extendNode_ptr = (struct Node*) ( (char*) root+(PGSIZE*extendLeafNode_offset) );

	bitmask_t passed_hIndex;
	bitmask_t last_hIndex;

	for( int n = block_init_position; n < mpiSEARCH; n += block_incremental_value )
	{
		passed_hIndex = 0;
		last_hIndex 	= root->branch[root->count-1].hIndex;

		struct Node* node_ptr 			= root;

		while( passed_hIndex < last_hIndex )
		{	
			//find out left most child node till leaf level
			while( node_ptr->level > 0 ) {

				if ( tid == 0) count[bid]++;

				__shared__ bool isOverlap;
				isOverlap = false;
				for ( int thread = tid ; thread < NODECARD ; thread+=NUMTHREADS){
					if(
							(thread < node_ptr->count) &&
							(node_ptr->branch[thread].hIndex > passed_hIndex) &&
							(devRectOverlap(&node_ptr->branch[thread].rect, &query[n])))
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

				/*
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
					 */
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
					passed_hIndex = node_ptr->branch[node_ptr->count-1].hIndex;

					if( tid == 0 ) rootCount[bid]++;
					node_ptr = root;
					break;
				}
				// there exists some overlapped node
				else{
					node_ptr = node_ptr->branch[ childOverlap[0] ].child;
				}
				__syncthreads();
			}


			while( node_ptr->level == 0 )
			{
				if ( tid == 0) count[bid]++;

				isHit = false;


				for ( int thread = tid ; thread < node_ptr->count; thread+= NUMTHREADS )
				{
					if ( devRectOverlap(&node_ptr->branch[thread].rect, &query[n]))
					{
						t_hit[thread]++;
						isHit = true;
					}
				}
				__syncthreads();

				passed_hIndex = node_ptr->branch[node_ptr->count-1].hIndex;

				//last leaf node

				if ( node_ptr->branch[node_ptr->count-1].hIndex == last_hIndex )
					break;

				else if( isHit )
				{
					node_ptr++;
				}
				else
				{
					node_ptr = extendNode_ptr + ( ( node_ptr - leafNode_ptr) / NODECARD) ;
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
