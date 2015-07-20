#include <host.h>
//#####################################################################
//####################### IMPLEMENTATIONS #############################
//#####################################################################


int hostRTreeSearch(struct Node *n, struct Rect *r , int *vNode)
{
	int hitCount = 0;
	int i;
	(*vNode)++;
	//printf("morton code %d\n",n->branch[0].hIndex);
	//printf("# of childs %d\n",n->count);
	//printf("level %d\n",n->level);

	if (n->level > 0) // this is an internal node in the tree //
	{
		for (i=0; i<n->count; i++)
			if (RectOverlap(r,&n->branch[i].rect))
			{
				hitCount += hostRTreeSearch(n->branch[i].child, r , vNode);
			}
	}
	else // this is a leaf node 
	{
		for (i=0; i<n->count; i++)
			if (RectOverlap(r,&n->branch[i].rect))
			{
				hitCount++;
			}
	}
	return hitCount;
}
int hostBVHTreeSearch(BVH_Node *n, struct Rect *r , int *vNode)
{
	int hitCount = 0;
	int i;
	(*vNode)++;
	//printf("morton code %d\n",n->branch[0].hIndex);
	//printf("# of childs %d\n",n->count);
	//printf("level %d\n",n->level);

	if (n->level > 0) // this is an internal node in the tree //
	{
		for (i=0; i<n->count; i++)
			if (RectOverlap(r,&n->branch[i].rect))
			{
				hitCount += hostBVHTreeSearch(n->branch[i].child, r , vNode);
			}
	}
	else // this is a leaf node 
	{
		for (i=0; i<n->count; i++)
			if (RectOverlap(r,&n->branch[i].rect))
			{
				hitCount++;
			}
	}
	return hitCount;
}

void* PThreadSearch(void* arg)
{
	int i;
	struct Rect r;
	struct thread_data* td = (struct thread_data*) arg;

	char queryFileName[100];


	if ( !strcmp(DATATYPE, "high"))
		sprintf(queryFileName, "/home/jwkim/inputFile/%s_dim_query.%d.bin.%ss", DATATYPE,NUMDIMS,SELECTIVITY);
	else
		sprintf(queryFileName, "/home/jwkim/inputFile/%s_dim_query.%d.bin.%ss.%s", DATATYPE,NUMDIMS,SELECTIVITY,querySize);



	FILE *fp = fopen(queryFileName, "r");
	if(fp==0x0){
		printf("Line %d : Pthread query file open error\n",__LINE__);
		exit(1);
	}

	fseek(fp, WORKLOAD*NUMSEARCH*sizeof(float)*2*NUMDIMS, SEEK_SET);

	for(i=0;i<NUMSEARCH;i++){
		for(int j=0;j<2*NUMDIMS;j++)
			fread(&r.boundary[j], sizeof(float), 1, fp);


		if(i%td->nblock==td->tid) { 
			// partition_cpu
			td->hit += hostRTreeSearch(td->root, &r, &td->vNode);

			if( i < (NUMSEARCH - NUMSEARCH%td->nblock) )
				pthread_barrier_wait(&bar);
		}

	}
	pthread_barrier_wait(&bar);
	if( fclose(fp) != 0 ){
		printf("Line %d : Pthread query file close error\n",__LINE__);
		exit(1);
	}

	pthread_exit(NULL);
}
void* PThreadSearch_BVH(void* arg)
{
	int i;
	struct Rect r;
	struct thread_data_BVH* td = (struct thread_data_BVH*) arg;

	char queryFileName[100];


	if ( !strcmp(DATATYPE, "high"))
		sprintf(queryFileName, "/home/jwkim/inputFile/%s_dim_query.%d.bin.%ss", DATATYPE,NUMDIMS,SELECTIVITY);
	else
		sprintf(queryFileName, "/home/jwkim/inputFile/%s_dim_query.%d.bin.%ss.%s", DATATYPE,NUMDIMS,SELECTIVITY,querySize);



	FILE *fp = fopen(queryFileName, "r");
	if(fp==0x0){
		printf("Line %d : Pthread query file open error\n",__LINE__);
		exit(1);
	}

	fseek(fp, WORKLOAD*NUMSEARCH*sizeof(float)*2*NUMDIMS, SEEK_SET);

	for(i=0;i<NUMSEARCH;i++){
		for(int j=0;j<2*NUMDIMS;j++)
			fread(&r.boundary[j], sizeof(float), 1, fp);


		if(i%td->nblock==td->tid) { 
			// partition_cpu
			td->hit += hostBVHTreeSearch(td->root, &r, &td->vNode);

			if( i < (NUMSEARCH - NUMSEARCH%td->nblock) )
				pthread_barrier_wait(&bar);
		}

	}
	pthread_barrier_wait(&bar);
	if( fclose(fp) != 0 ){
		printf("Line %d : Pthread query file close error\n",__LINE__);
		exit(1);
	}

	pthread_exit(NULL);
}

void RTree_Multicore()
{
	struct timeval t1, t2;
	//cudaEvent_t start_event, stop_event;
	//cudaEventCreate(&start_event);
	//cudaEventCreate(&stop_event);
	float elapsed_time;

	void *status;
	long hit = 0; // Initialize host hit counter
	long visitedNodes = 0;

	int rc;

	pthread_t threads[NCPUCORES];
	struct thread_data td[NCPUCORES];

	pthread_barrier_init(&bar, NULL, NCPUCORES);

	//cudaEventRecord(start_event, 0);
	gettimeofday(&t1,0); // start the stopwatch

	// Using PThreads for searching
	for(int i=0; i<NCPUCORES; i++){

		td[i].tid = i;
		//td[i].tid = 0; // partition_cpu

		td[i].nblock = NCPUCORES;
		td[i].hit = 0;
		td[i].vNode = 0;

		if( PARTITIONED == 1 )
			td[i].root = (struct Node*)ch_root[0];
		else
			td[i].root = (struct Node*)ch_root[i];

		rc = pthread_create(&threads[i], NULL, PThreadSearch, (void *)&td[i]);
		if (rc){
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

	for(int i=0; i<NCPUCORES; i++) {
		rc = pthread_join(threads[i], &status);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
		hit += td[i].hit;
		visitedNodes += td[i].vNode;
	}
	pthread_barrier_destroy(&bar);

	//cudaEventRecord(stop_event, 0) ;
	//cudaEventSynchronize(stop_event) ;
	//cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	gettimeofday(&t2,0); // stop the stopwatch
	elapsed_time = (t2.tv_sec-t1.tv_sec)*1000000 + (t2.tv_usec - t1.tv_usec);
	/*
		 elapsed_time /= 1000;

		 printf("RTree-Multicore time          : %.3f ms\n", elapsed_time);
		 printf("RTree-Multicore HIT           : %lu\n",hit);
		 */

	float selectionRatio = ((float)(hit/NUMSEARCH)/NUMDATA);

	printf("-----HOST RTree experiments Result-----\n");
	printf("  	HOST Total Hit  = %lu   hits\n", hit);
	printf("  	Selection ratio = %.5f % \n",(float)selectionRatio*100);
	printf("  	Visit Counter   = %lu  nodes per query\n", visitedNodes/NUMSEARCH);
	printf("  	Time measured   =  %f  ms\n", elapsed_time/1000);
	printf("--------HOST experiments Result--------\n\n\n");




	fflush(stdout);
}
void BVHTree_Multicore()
{
	struct timeval t1, t2;
	//cudaEvent_t start_event, stop_event;
	//cudaEventCreate(&start_event);
	//cudaEventCreate(&stop_event);
	float elapsed_time;

	void *status;
	long hit = 0; // Initialize host hit counter
	long visitedNodes = 0;

	int rc;

	pthread_t threads[NCPUCORES];
	struct thread_data_BVH td[NCPUCORES];

	pthread_barrier_init(&bar, NULL, NCPUCORES);

	//cudaEventRecord(start_event, 0);
	gettimeofday(&t1,0); // start the stopwatch

	// Using PThreads for searching
	for(int i=0; i<NCPUCORES; i++){

		td[i].tid = i;
		//td[i].tid = 0; // partition_cpu

		td[i].nblock = NCPUCORES;
		td[i].hit = 0;
		td[i].vNode = 0;

		if( PARTITIONED == 1 )
			td[i].root = (BVH_Node*)ch_root[0];
		else
			td[i].root = (BVH_Node*)ch_root[i];

		rc = pthread_create(&threads[i], NULL, PThreadSearch_BVH, (void *)&td[i]);
		if (rc){
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

	for(int i=0; i<NCPUCORES; i++) {
		rc = pthread_join(threads[i], &status);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
		hit += td[i].hit;
		visitedNodes += td[i].vNode;
	}
	pthread_barrier_destroy(&bar);

	//cudaEventRecord(stop_event, 0) ;
	//cudaEventSynchronize(stop_event) ;
	//cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	gettimeofday(&t2,0); // stop the stopwatch
	elapsed_time = (t2.tv_sec-t1.tv_sec)*1000000 + (t2.tv_usec - t1.tv_usec);
	/*
		 elapsed_time /= 1000;

		 printf("RTree-Multicore time          : %.3f ms\n", elapsed_time);
		 printf("RTree-Multicore HIT           : %lu\n",hit);
		 */

	float selectionRatio = ((float)(hit/NUMSEARCH)/NUMDATA);

	printf("----HOST BVHTree experiments Result----\n");
	printf("  	HOST Total Hit  = %lu   hits\n", hit);
	printf("  	HOST Total Hit  = %lu   hits\n", hit);
	printf("  	Selection ratio = %.5f % \n",(float)selectionRatio*100);
	printf("  	Visit Counter   = %lu  nodes per query\n", visitedNodes/NUMSEARCH);
	printf("  	Time measured   =  %f  ms\n", elapsed_time/1000);
	printf("--------HOST experiments Result--------\n\n\n");




	fflush(stdout);
}
