//#####################################################################
//########################### MPES ####################################
//#####################################################################


void MPES(int number_of_procs, int myrank);
__global__ void globalMPES(struct Rect * query, int *b_hit, long * d_leafNode_offset, int* d_numberOfLeafnodes,  int mpiSEARCH, int boundary_index, int PARTITIONED);
__global__ void globalMPES_ILP(struct Rect * query, int *b_hit, long * d_leafNode_offset, int* d_numberOfLeafnodes,  int mpiSEARCH, int boundary_index, int PARTITIONED);


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
	}


	long leafNode_offset[2];
	leafNode_offset[0] = (indexSize[0]/PGSIZE) - number_of_node_in_level[0][0];
	leafNode_offset[1] = (indexSize[1]/PGSIZE) - number_of_node_in_level[1][0];

	int numberOfLeafnodes[2];
	numberOfLeafnodes[0] = number_of_node_in_level[0][0]; 
	numberOfLeafnodes[1] = number_of_node_in_level[1][0];

	long* d_leafNode_offset;
	cudaMalloc((void**)&d_leafNode_offset, sizeof(long)*2);
	cudaMemcpy( d_leafNode_offset, leafNode_offset, sizeof(long)*2, cudaMemcpyHostToDevice);

	int* d_numberOfLeafnodes;
	cudaMalloc((void**)&d_numberOfLeafnodes, sizeof(int)*2);
	cudaMemcpy( d_numberOfLeafnodes, numberOfLeafnodes, sizeof(int)*2, cudaMemcpyHostToDevice);

	int h_hit[NUMBLOCK];
	long t_hit = 0;
	int* d_hit;
	cudaMalloc( (void**) &d_hit,   NUMBLOCK*sizeof(int) );


#ifdef MPI
	int mpiSEARCH = NUMSEARCH/number_of_procs;
	if( myrank < NUMSEARCH%number_of_procs )
		mpiSEARCH++;

	struct Rect* d_query;
	cudaMalloc( (void**) &d_query, mpiSEARCH*sizeof(struct Rect) );

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

#ifndef ILP
	globalMPES<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_leafNode_offset, d_numberOfLeafnodes,  mpiSEARCH, boundary_of_trees, PARTITIONED);
#else
	globalMPES_ILP<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, d_leafNode_offset, d_numberOfLeafnodes,  mpiSEARCH, boundary_of_trees, PARTITIONED);
#endif

	// Copy hit count from device to host
	cudaMemcpy(h_hit, d_hit, NUMBLOCK*sizeof(int), cudaMemcpyDeviceToHost);
	for(int j=0;j<NUMBLOCK;j++){
		t_hit += h_hit[j];
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
			globalMPES<<<nBatch, NODECARD>>>(d_query, d_hit, d_leafNode_offset, d_numberOfLeafnodes,  nBatch, boundary_of_trees, PARTITIONED);
#else
			globalMPES_ILP<<<nBatch, NUMTHREADS>>>(d_query, d_hit, d_leafNode_offset, d_numberOfLeafnodes,  nBatch, boundary_of_trees, PARTITIONED);
#endif
			cudaMemcpy(h_hit, d_hit, nBatch*sizeof(int), cudaMemcpyDeviceToHost);

			for(int j=0;j<nBatch;j++){
				t_hit += h_hit[j];
			}

		}

		cudaThreadSynchronize();
		cudaEventRecord(stop_event, 0) ;
		cudaEventSynchronize(stop_event) ;

		cudaFree( d_query);
	}else
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
			globalMPES<<<NUMBLOCK, NODECARD>>>(d_query, d_hit, d_leafNode_offset, d_numberOfLeafnodes, 1, boundary_of_trees, PARTITIONED);
#else
			globalMPES_ILP<<<NUMBLOCK, NUMTHREADS>>>(d_query, d_hit, d_leafNode_offset, d_numberOfLeafnodes,  1, boundary_of_trees, PARTITIONED);
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
#endif 


	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	printf("MPES time          : %.3f ms\n", elapsed_time);
	printf("MPES HIT           : %lu\n",t_hit);
	//selection ratio
#ifdef MPI
	printf("MPES HIT ratio(%)  : %5.5f\n",((t_hit/mpiSEARCH)/(float)(NUMDATA/100)));
#else
	printf("MPES HIT ratio(%)  : %5.5f\n",((t_hit/NUMSEARCH)/(float)(NUMDATA/100)));
#endif

	float MIN_time, MAX_time;
	MPI_Reduce(&elapsed_time, &MIN_time, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&elapsed_time, &MAX_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
	if( myrank == 0 ){
		int MAX_THROUGHPUT, MIN_THROUGHPUT;
		MAX_THROUGHPUT = ( NUMSEARCH / ( MAX_time/1000));
		MIN_THROUGHPUT = ( NUMSEARCH / ( MIN_time/1000));
		printf("MIN %d MAX %d\n", MAX_THROUGHPUT, MIN_THROUGHPUT);

		//printf("MAX %.3f MIN %.3f\n", MAX_time, MIN_time);
	}

	fclose(fp);

	cudaFree( d_leafNode_offset); 
	cudaFree( d_numberOfLeafnodes); 
	cudaFree( d_hit); 
}



__global__ void globalMPES(struct Rect * query, int *b_hit, long * d_leafNode_offset, int* d_numberOfLeafnodes,  int mpiSEARCH, int boundary_index, int PARTITIONED)
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
	b_hit[bid] = 0;
	t_hit[tid] = 0;

	__syncthreads();

	struct Node* root = deviceRoot[partition_index]+leafNode_offset;

	for( int n = block_init_position; n < mpiSEARCH ; n += block_incremental_value ) 
	{
		struct Node* node = root;

		for(int i=0; i<numberOfLeafnodes; i++ )
		{
			if( tid < node->count)
			{
				if( devRectOverlap(&query[n], &node->branch[tid].rect ))
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

__global__ void globalMPES_ILP(struct Rect * query, int *b_hit, long * d_leafNode_offset, int* d_numberOfLeafnodes,  int mpiSEARCH, int boundary_index, int PARTITIONED)
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
	b_hit[bid] = 0;


	for(int thread = tid; thread < NODECARD; thread+= NUMTHREADS)
		t_hit[thread] = 0;
	__syncthreads();

	struct Node* root = deviceRoot[partition_index]+leafNode_offset;

	for( int n = block_init_position; n < mpiSEARCH ; n += block_incremental_value ) 
	{
		struct Node* node = root;

		for(int i=0; i<numberOfLeafnodes; i++  )
		{
			for(int thread = tid; thread < NODECARD ; thread+= NUMTHREADS)
			{
				if( (thread < node->count) && (devRectOverlap(&query[n], &node->branch[thread].rect )))
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

