#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include "hilbert.h"

#include <float.h>
#include <math.h>
#include "sys/time.h"

//for cuda checking
#include <assert.h>

//MPI
#include <mpi.h>
using namespace std;



#define NUMDIMS	8		     // number of dimensions
#define DATATYPE "high"  // For NOAA real datasets, number of dimensions shoulde be 3
#define NODECARD 256     // number of nodecard , it MUST be square root of 2.

#define NUMTHREADS 64

#define PGSIZE	(int) ( (sizeof(int)*2) + ( NODECARD*sizeof(struct Branch) ) ) // automatically calculate the pagesize 

//##########################################################################
//########################### GLOBAL VARIABLES #############################
//##########################################################################

//MPI
int number_of_procs = 1 , myrank = 0;

//Arguments
int NUMDATA, NUMBLOCK, NUMSEARCH, PARTITIONED, WORKLOAD, NCPUCORES, POLICY;
char querySize[20], SELECTIVITY[20];

//roots for cpu
char** ch_root;
char** cd_root; // store leaf nodes temporarily

int indexSize[2];
int boundary_of_trees;
int tree_height[2];
int *number_of_node_in_level[2];


//##########################################################################
//############################ STRUCTURES ##################################
//##########################################################################

struct Rect //8*NUMDIMS
{
	float boundary[2*NUMDIMS]; // float (4byte)*2*NUMDIMS = 8*NUMDIMS(byte)
};
struct Node;
struct Branch  //  16 + 8*NUMDIMS (byte)
{
	struct Rect rect;		// 8 + 8*NUMDIMS (byte)
	bitmask_t hIndex; // unsigned long long (8byte)
	struct Node *child; // Pointer (8byte)
};

struct Node 
{
	int count; // int(4byte)
	int level; // int(4byte) /* 0 is leaf, others positive */
	struct Branch branch[NODECARD]; 
};

struct thread_data{
	int  tid;
	int  partition_no;
	long  hit;
	struct Node* root;
};

pthread_barrier_t bar;

//#####################################################################
//################## INSERTION and SORT FUNCTIONS #####################
//#####################################################################

bool ParseArgs(int argc, char **argv);
void InsertData(bitmask_t* keys, struct Branch* data);
int HilbertCompare(const void* a, const void* b);

bool TreeDumpToFile();
bool TreeLoadFromFile();
void TreeLoadFromMem(struct Node* n, char *buf);


//#####################################################################
//######################## BUILD FUNCTIONS ############################
//#####################################################################

int print_index(struct Node* node, int tNODE );

void Bulk_loading( struct Branch* data );
/*more than one means partitioned version and task parallelism );*/
//#####################################################################
//###################### SEARCH FUNCTIONS #############################
//#####################################################################


bool  RectOverlap(struct Rect *r, struct Rect *s);
__device__ bool devRectOverlap(struct Rect *r, struct Rect *s);

void searchDataOnCPU();
int hostLinearSearchData(struct Branch* b, struct Rect* r );
int hostRTreeSearch(struct Node *n, struct Rect *r );
void* PThreadSearch(void* arg);
void RTree_multicore();


void MPES(int number_of_procs, int myrank);

void MPTS(int number_of_procs, int myrank);

void MPHR(int number_of_procs, int myrank);

void MPHR2(int number_of_procs, int myrank);

//#####################################################################
//####################### IMPLEMENTATIONS #############################
//#####################################################################


	inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n",
				cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

void profileCopies(float        *h_a,
		float        *h_b,
		float        *d,
		unsigned int  n,
		char         *desc)
{
	printf("\n%s transfers\n", desc);

	unsigned int bytes = n * sizeof(float);

	// events for timing
	cudaEvent_t startEvent, stopEvent;

	checkCuda( cudaEventCreate(&startEvent) );
	checkCuda( cudaEventCreate(&stopEvent) );

	checkCuda( cudaEventRecord(startEvent, 0) );
	checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );

	float time;
	checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

	checkCuda( cudaEventRecord(startEvent, 0) );
	checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );

	checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

	for (int i = 0; i < n; ++i) {
		if (h_a[i] != h_b[i]) {
			printf("*** %s transfers failed ***", desc);
			break;
		}
	}
	checkCuda( cudaEventDestroy(startEvent) );
	checkCuda( cudaEventDestroy(stopEvent) );
}



bool ParseArgs(int argc, char **argv) {

	//!!! add the policy (eg., round robin, dema, etc..)

	//assign default values
	NUMDATA = 0;
	NUMSEARCH = 0;
	PARTITIONED = 1; // 1 : braided version, over than 1 : partitioned version
	NUMBLOCK = 128; 
	strcpy(SELECTIVITY , "1"); // 1 %
	NCPUCORES = 1;  
	WORKLOAD = 0;
	POLICY = 0;

	static const char *options="d:D:q:Q:p:P:b:B:s:S:c:C:w:W:o:O:";
	extern char *optarg;
	int c;

	while ((c = getopt(argc, argv, options)) != -1) {
		switch (c) {
			case 'd':
			case 'D': NUMDATA = atoi(optarg); break;
			case 'q':
			case 'Q': NUMSEARCH = atoi(optarg); break;
			case 'p':
			case 'P': NUMBLOCK = PARTITIONED = atoi(optarg); break;
			case 'b':
			case 'B': NUMBLOCK = atoi(optarg); PARTITIONED=1; break;
			case 's':
			case 'S': strcpy(SELECTIVITY , optarg); break;
			case 'c':
			case 'C': NCPUCORES = atoi(optarg); break;
			case 'w':
			case 'W': WORKLOAD = atoi(optarg); break;
			case 'o':
			case 'O': POLICY = atoi(optarg); break;

			default: break;
		} // end of switch
	} // end of while

	ch_root = (char**)malloc(PARTITIONED*sizeof(char*));
	cd_root = (char**)malloc(PARTITIONED*sizeof(char*));

	if( NUMDATA == 1000000 )
		strcpy( querySize,"1m");
	else if( NUMDATA == 2000000 )
		strcpy( querySize,"2m");
	else 	if( NUMDATA == 4000000 )
		strcpy( querySize,"4m");
	else 	if( NUMDATA == 8000000 )
		strcpy( querySize,"8m");
	else 	if( NUMDATA == 16000000 )
		strcpy( querySize,"16m");
	else 	if( NUMDATA == 32000000 )
		strcpy( querySize,"32m");
	else 	if( NUMDATA == 40000000 )
		strcpy( querySize,"40m");

	printf("DATATYPE : %s, PGSIZE %d, NUMDIMS %d, NUMBLOCK %d, NUMTHREADS %d,  NODECARD %d, NUMDATA %d, NUMSEARCH %d, SELECTION RATIO %s, NCPU %d,   PARTITIONED %d, ",
			DATATYPE,      PGSIZE,    NUMDIMS,    NUMBLOCK,    NUMTHREADS,     NODECARD,    NUMDATA,    NUMSEARCH,    SELECTIVITY ,       NCPUCORES, PARTITIONED       );
	if( POLICY == 0 )
		printf("Original distribution\n");
	else
		printf("Roundrobin distribution\n");


	return true;
}


void InsertData(bitmask_t* keys, struct Branch* data){

	char dataFileName[100];

	if( !strcmp(DATATYPE, "high" ) )
		sprintf(dataFileName, "/home/jwkim/inputFile/%s_dim_data.%d.bin\0", DATATYPE, NUMDIMS); 
	else
		sprintf(dataFileName, "/home/jwkim/inputFile/NOAA%d.bin",myrank); 

	//sprintf(dataFileName, "/tmp/ramdisk0/NOAA.bin"  ); 

	FILE *fp = fopen(dataFileName, "r");
	if(fp==0x0){
		printf("Line %d : Insert file open error\n", __LINE__);
		printf("%s\n",dataFileName);
		exit(1);
	}

	//!!! nBits*NUMDIMS can not equal and larger than 64(bitmask_t)
	bitmask_t nBits = 64/NUMDIMS; //64 is maximum dimensions

	struct Branch b;
	for(int i=0; i<NUMDATA; i++){

		bitmask_t coord[NUMDIMS];
		for(int j=0;j<NUMDIMS;j++){
			fread(&b.rect.boundary[j], sizeof(float), 1, fp);
			b.rect.boundary[NUMDIMS+j] = b.rect.boundary[j];
			if( !strcmp(DATATYPE, "high"))
				coord[j] = (bitmask_t)(1000*b.rect.boundary[j]);
			else
				coord[j] = (bitmask_t)(1000000*b.rect.boundary[j]);
		}

		//synthetic
		if( !strcmp(DATATYPE, "high"))
			b.hIndex = hilbert_c2i(NUMDIMS-1, nBits, coord);
		else //real datasets from NOAA
			b.hIndex = hilbert_c2i(NUMDIMS, 20, coord);

		keys[i] = b.hIndex;
		data[i] = b;
	}

	if(fclose(fp) != 0){
		printf("Line %d : Insert file close error\n", __LINE__);
		exit(1);
	}
}

// Hibert Comparing Function
int HilbertCompare(const void* a, const void* b)
{
	struct Branch* b1 = (struct Branch*) a;
	struct Branch* b2 = (struct Branch*) b;

	return (b1->hIndex > b2->hIndex) ? 1 : 0;
}

void TreeLoadFromMem(struct Node* n, char *buf)
{
	int i;

	if (n->level > 0) // this is an internal node in the tree 
	{
		//printf("node's level %d \n", n->level);
		for (i=0; i<n->count; i++) {
			// translate file offset to memory pointer
			n->branch[i].child = (struct Node*) ((size_t) n->branch[i].child + (size_t) buf);
			TreeLoadFromMem(n->branch[i].child, buf);
		}
	}

	return;
}

int print_index(struct Node* node, int tNODE )
{
	int nNODE = 0;

	for( int i = 0 ; i < tNODE  ; i ++, node++ ){
		//{DEBUG
		//printf(" # %d || %d  ",i,node);
		cout << "node " << node << endl;
		printf("count  %d, ", node->count);
		printf("level  %d\n", node->level);

		for( int c = 0 ; c < node->count; c ++)
		{
			for( int d = 0; d < NUMDIMS*2 ; d ++)
			{
				printf(" dims : %d rect %5.5f \n",d , node->branch[c].rect.boundary[d]);
			}
			printf("hilbert value %lu\n",node->branch[c].hIndex);
			//printf("child %d\n",node->branch[c].child);
			//cout << "child " << c << " : " << (long)node->branch[c].child << endl;

			//printf(" %d\n", node->branch[c].child);
		}
		//}DEBUG
		//if( node->level > 0 )
		//	nNODE += node->count;
	}

	return nNODE;
}


	void Bulk_loading( struct Branch* data )
	{
		cudaEvent_t start_event, stop_event;
		checkCuda ( cudaEventCreate(&start_event) );
		checkCuda ( cudaEventCreate(&stop_event) );

		//measure memory transfer time btw host and device
		cudaEvent_t memcpy_start, memcpy_stop;
		checkCuda ( cudaEventCreate(&memcpy_start) );
		checkCuda ( cudaEventCreate(&memcpy_stop) );

		float elapsed_time;
		float memcpy_elapsed_time;

		//###########################################################################
		//################### ATTACH THE LEAF NODE ON THE INDEX #####################
		//###########################################################################

		checkCuda ( cudaEventRecord(start_event, 0) );

		boundary_of_trees = NUMDATA%PARTITIONED; // the boundary number which that distinguish trees by tree height

		// there are trees which have different tree heights
		int partitioned_NUMDATA[2];
		partitioned_NUMDATA[0] = partitioned_NUMDATA[1] = (NUMDATA/PARTITIONED);
		if( boundary_of_trees )
			partitioned_NUMDATA[0]++;

		// if number of data is not exactly divide by number of block,
		tree_height[0] = ceil ( log( partitioned_NUMDATA[0] ) / log( NODECARD ) ) ;
		tree_height[1] = ceil ( log( partitioned_NUMDATA[1] ) / log( NODECARD ) );

		//DEBUGGING
		printf("Tree height 0 is %d\n",tree_height[0]);
		printf("Tree height 1 is %d\n",tree_height[1]);


		//int number_of_node_in_level[2][ tree_height[0] ] ;
		number_of_node_in_level[0] = (int*)malloc(sizeof(int)*tree_height[0]);
		number_of_node_in_level[1] = (int*)malloc(sizeof(int)*tree_height[0]);

		number_of_node_in_level[0][0] = ceil ( (double) partitioned_NUMDATA[0]/NODECARD  )  ;
		number_of_node_in_level[1][0] = ceil ( (double) partitioned_NUMDATA[1]/NODECARD  )  ;

		indexSize[0] = indexSize[1] = 0;

		for ( int i = 0 ; i < tree_height[1]-1 ; i ++ )
		{
			number_of_node_in_level[0][i+1] = ceil ( (double)number_of_node_in_level[0][i]/NODECARD ) ;
			number_of_node_in_level[1][i+1] = ceil ( (double)number_of_node_in_level[1][i]/NODECARD ) ;
		}
		if( tree_height[0] > tree_height[1] ){
			number_of_node_in_level[0][tree_height[0]-1] = ceil ( (double)number_of_node_in_level[0][tree_height[0]-2]/NODECARD )  ;
			number_of_node_in_level[0][tree_height[0]-1] = 0;
		}

		for ( int i = tree_height[1]-1 ; i >= 0 ; i -- )
		{
			//printf("level : %d number of node : %d \n",i, number_of_node_in_level[partition_no][i]);
			indexSize[0] += number_of_node_in_level[0][i]*PGSIZE;
			indexSize[1] += number_of_node_in_level[1][i]*PGSIZE;
		}
		if( tree_height[0] > tree_height[1] )
			indexSize[0] += number_of_node_in_level[0][tree_height[0]-1]*PGSIZE;
		int partitioned_offset = 0 ;

		int tree_idx = 0;
		for ( int partition_no = 0 ; partition_no < PARTITIONED; partition_no++)
		{
			if( partition_no == boundary_of_trees )
				tree_idx = 1;

			//pageable memory
			//ch_root[partition_no] = (char*) malloc ( indexSize[tree_idx] );
			//cd_root[partition_no] = (char*) malloc ( indexSize[tree_idx] );
			//pinned memory
			checkCuda( cudaMallocHost( (void**)& ch_root[partition_no] , indexSize[tree_idx] ));
			checkCuda( cudaMallocHost( (void**)& cd_root[partition_no] , indexSize[tree_idx] ));

			//attach the leafnodes to the memory buffer
			int count_and_level[2] = { NODECARD /* count */ , 0 /* level */ };
			long offset = indexSize[tree_idx] - ( number_of_node_in_level[tree_idx][0] * PGSIZE ) ;

			for( int i = 0 ; i < number_of_node_in_level[tree_idx][0]-1 ; i ++ )
			{
				memcpy(ch_root[partition_no]+offset,   count_and_level,   sizeof(int)*2 );
				memcpy(ch_root[partition_no]+offset+8, data + NODECARD*i + partitioned_offset, sizeof(struct Branch)*NODECARD);
				offset += PGSIZE ;
			}

			//last node in leaf level, change the count if necessary
			if( partitioned_NUMDATA[tree_idx] % NODECARD )
				count_and_level[0] = partitioned_NUMDATA[tree_idx] % NODECARD; // count
			memcpy(ch_root[partition_no]+offset, count_and_level, sizeof(int)*2 );
			memcpy(ch_root[partition_no]+offset+8, data + NODECARD * (number_of_node_in_level[tree_idx][0]-1) + partitioned_offset , sizeof(struct Branch)*count_and_level[0]); 

			memcpy(cd_root[partition_no], ch_root[partition_no], indexSize[tree_idx]);

			partitioned_offset += partitioned_NUMDATA[tree_idx];

			//DEBUGGING
			//print leaves for debugging
			//offset = indexSize[tree_idx] - ( number_of_node_in_level[tree_idx][0] * PGSIZE ) ;
			//struct Node* temp = (struct Node*)(ch_root[partition_no]+offset);
			//printf("PRINT LEAVES for DEBUGGING\n");
			//print_index(temp, number_of_node_in_level[tree_idx][0]);

		} 


		checkCuda ( cudaEventRecord(stop_event, 0) );
		checkCuda ( cudaEventSynchronize(stop_event) );
		checkCuda ( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
		printf("Attaching Time = %.3fs\n\n", elapsed_time/1000.0f);


		//###########################################################################
		//######################## BUILD-UP USING CPU ###############################
		//###########################################################################

#if BUILD_ON_CPU

		tree_idx = 0;
		checkCuda ( cudaEventRecord(start_event, 0) );
		for ( int partition_no = 0; partition_no < PARTITIONED ; partition_no ++ )
		{
			if( partition_no == boundary_of_trees)
				tree_idx = 1;

			char* root = ch_root[partition_no];
			struct Node* ptr_node, *parent_node ;
			int offset = indexSize[tree_idx];

			for( int level = 0 ; level < tree_height[tree_idx]-1 ; level ++ )
			{
				//start offset of each level
				offset -=  number_of_node_in_level[tree_idx][level] * PGSIZE ;

				ptr_node    = (struct Node*) ( (char*)root + offset ) ;
				parent_node = (struct Node*) ( (char*)root + ( offset - ( number_of_node_in_level[tree_idx][level+1] * PGSIZE ))) ;

				for( int n = 0 ; n < number_of_node_in_level[tree_idx][level] ; n ++ , ptr_node++)
				{
					//move to next parent node
					if( !( n % NODECARD ) ){
						if( n ) parent_node++;
						parent_node->level = level+1;
						parent_node->count = 0;
					}

					//set the child pointer of parent node to this node
					parent_node->branch[parent_node->count].child = ptr_node;

					//Find out the min, max boundaries in this node and set up the parent rect.
					float MIN_boundary = FLT_MAX;
					float MAX_boundary = FLT_MIN;
					for( int d = 0 ; d < NUMDIMS ; d ++ )
					{
						MIN_boundary = FLT_MAX;
						MAX_boundary = FLT_MIN;

						for( int c = 0 ; c < ptr_node->count ; c ++ )
						{
							if( MIN_boundary > ptr_node->branch[c].rect.boundary[d] )
								MIN_boundary = ptr_node->branch[c].rect.boundary[d];
							if( MAX_boundary < ptr_node->branch[c].rect.boundary[d+NUMDIMS] )
								MAX_boundary = ptr_node->branch[c].rect.boundary[d+NUMDIMS];
						}
						parent_node->branch[parent_node->count].rect.boundary[d]
							= MIN_boundary;
						parent_node->branch[parent_node->count].rect.boundary[d+NUMDIMS]
							= MAX_boundary;
					}

					//set the parent branch hilbert index
					parent_node->branch[parent_node->count].hIndex = ptr_node->branch[ptr_node->count-1].hIndex;
					//increaes count of parent node
					parent_node->count++;

				}
			}
		}

		printf("Build Time on CPU = %.3fs\n\n", elapsed_time/1000.0f);
#endif

		//###########################################################################
		//######################## BUILD-UP USING GPU ###############################
		//###########################################################################

#endif

	}


	//#####################################################################
	//###################### SEARCH FUNCTIONS #############################
	//#####################################################################


	bool  RectOverlap(struct Rect *r, struct Rect *s)
	{
		int i, j;

		for (i=0; i<NUMDIMS; i++)
		{
			j = i + NUMDIMS;  

			if (r->boundary[i] > s->boundary[j] || s->boundary[i] > r->boundary[j])
			{
				return false;
			}
		}
		return true;
	}
	

	void searchDataOnCPU()
	{
	//Open query file
	char queryFileName[100];

	if ( !strcmp(DATATYPE, "high"))
	sprintf(queryFileName, "/home/jwkim/inputFile/%s_dim_query.%d.bin.%ss", DATATYPE,NUMDIMS,SELECTIVITY);
	else
	sprintf(queryFileName, "/home/jwkim/inputFile/%s_dim_query.%d.bin.%ss.%s", DATATYPE,NUMDIMS,SELECTIVITY,querySize);



	FILE *fp = fopen(queryFileName, "r");
	if(fp==0x0){
	printf("Line %d : query file open error\n",__LINE__);
	exit(1);
	}


	struct Rect r;

	//int hit = 0;
#pragma openmp 
	for( int i = 0 ; i < NUMSEARCH; i++){
		for(int j=0;j<2*NUMDIMS;j++){
			fread(&r.boundary[j], sizeof(float), 1, fp);
		}

		//hit += hostLinearSearchData(b, &r);
		hit += hostRTreeSearch(root, &r, &nNODE);
	}

//cout << "linear search    hit is " << hit << endl;
//cout << "host Rtree search  hit    is " << hit << endl;
//cout << "host Rtree search  vcount is " << nNODE << endl;

fclose(fp);
}
/*
	 int hostLinearSearchData(struct Branch* b, struct Rect* r )
	 {

	 int hit = 0;

	 for( int i = 0 ; i < NUMDATA; i++) 
	 {
	 if( RectOverlap(&b[i].rect, r) )
	 {
	 hit++;
//printf("Hit leaf node's position is %d\n", i/NODECARD);
}
}

return hit;
}
*/

int hostRTreeSearch(struct Node *n, struct Rect *r )
{
	int hitCount = 0;
	int i;

	if (n->level > 0) // this is an internal node in the tree //
	{
		for (i=0; i<n->count; i++)
			if (RectOverlap(r,&n->branch[i].rect))
			{
				hitCount += hostRTreeSearch(n->branch[i].child, r );
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

/*
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
	 td->hit += hostRTreeSearch(td->root, &r);

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
	 */

void RTree_multicore()
{
	/*
		 struct timeval t1, t2;
	//cudaEvent_t start_event, stop_event;
	//cudaEventCreate(&start_event);
	//cudaEventCreate(&stop_event);
	float elapsed_time;

	void *status;
	long hit = 0; // Initialize host hit counter

	int rc;
	pthread_t threads[NCPUCORES];
	struct thread_data td[NCPUCORES];

	pthread_barrier_init(&bar, NULL, NCPUCORES);

	//cudaEventRecord(start_event, 0);
	gettimeofday(&t1,0); // start the stopwatch

	// Using PThreads for searching
	for(int i=0; i<NCPUCORES; i++){
	td[i].tid = i;
	td[i].nblock = NCPUCORES;
	td[i].hit = 0;
	td[i].root = hostRoot;
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
	}
	pthread_barrier_destroy(&bar);

	//cudaEventRecord(stop_event, 0) ;
	//cudaEventSynchronize(stop_event) ;
	//cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

	gettimeofday(&t2,0); // stop the stopwatch
	elapsed_time = (t2.tv_sec-t1.tv_sec)*1000000 + (t2.tv_usec - t1.tv_usec);
	elapsed_time /= 1000;

	printf("RTree-multicore time          : %.3f ms\n", elapsed_time);
	printf("RTree-multicore HIT           : %lu\n",hit);

	//selection ratio
	printf("%5.5f\n",((hit/NUMSEARCH)/(float)(NUMDATA/100)));
	printf("THROUGHPUT %d\n",( NUMSEARCH / ( elapsed_time / 1000 ) ));


	fflush(stdout);
	*/
}

