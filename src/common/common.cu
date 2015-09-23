#include <common.h>


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



void printHelp(char **argv)
{
	std::cerr << "Usage:\n" << *argv << endl << 
			" -d number of data\n" 
			" [ -q number of queries, default : 0 (insertion mode) ]\n" 
			" [ -i index type, default : R-trees]\n"
			" [ -m search algorithm type, 1: MPES, 2: MPTS, 3: MPHR, 4: MPHR2\n \
			                            5: Short-Stack, 6: Parent-Link, 7: Skip-Pointer ]\n"
			" [ -o distribution policy, default : braided version]\n"
			" [ -b braided version, number of block, default : 128 ]\n"
			" [ -p partitioned version, number of block ]\n" 
			" [ -s selection ratio(%), default : 1 (%) ]\n"
			" [ -c number of cpu cores, default : 1 ]\n" 
			" [ -w workload offset, default : 0 ]\n" 
			" [ -v Specified device(GPU) id, default : 0 ]\n" 
			"\n e.g:   ./cuda -d 1000000 -q 1000 -s 0.5 -c 4 -w 3\n" 
			<< std::endl;
}

bool ParseArgs(int argc, char **argv) 
{

	//!!! add the policy (eg., round robin, dema, etc..)

	//assign default values
	NUMDATA = 0;
	NUMSEARCH = 0;
	PARTITIONED = 1; // 1 : braided version, over than 1 : partitioned version
	NUMBLOCK = 128; 
	strcpy(SELECTIVITY , "0.01"); // 0.05 %
	NCPUCORES =  0;
	WORKLOAD = 0;
	POLICY = 0;
	BUILD_TYPE = 0;
	DEVICE_ID = 0;
	keepDoing = 0;

	for( int i=0; i<7; i++)
	{
		t_time[i] = 0.0f;
		t_visit[i] = .0f;
		t_rootVisit[i] = .0f;
		t_pop[i] = .0f;
		t_push[i] = .0f;
		t_parent[i] = .0f;
		t_skipPointer[i] = .0f;
		METHOD[i] = false;
	}
	METHOD[7] = false;

	static const char *options="d:D:q:Q:p:P:b:B:s:S:c:C:w:W:o:O:i:I:m:M:k:K:v:V:";
	extern char *optarg;
	int c;
	char char_NUMDATA[20];

	while ((c = getopt(argc, argv, options)) != -1) {
		switch (c) {
			case 'v':
			case 'V': DEVICE_ID = atoi(optarg); break;
			case 'k':
			case 'K': keepDoing = atoi(optarg); break;
			case 'd':
			case 'D': strcpy(char_NUMDATA,optarg); break;
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
			case 'i':
			case 'I': BUILD_TYPE = atoi(optarg); break;
			case 'm':
			case 'M': METHOD[atoi(optarg)-1] = true;
								optind--;
								for( ;optind < argc && *argv[optind] != '-'; optind++)
								{
									METHOD[atoi( argv[optind] )-1] = true;
								}
								break;
			default: break;
		} // end of switch
	} // end of while

	if( NCPUCORES > 0 )
		NCPUCORES = ( PARTITIONED > 1 ) ? PARTITIONED : 1;  

	ch_root = (char**)malloc(PARTITIONED*sizeof(char*));
	cd_root = (char**)malloc(PARTITIONED*sizeof(char*));


	if( char_NUMDATA[strlen(char_NUMDATA)-1] == 'm')
	{
    char_NUMDATA[strlen(char_NUMDATA)-1] =  '\n';
		NUMDATA = atoi(char_NUMDATA)*1000000;
	}
	else 
		NUMDATA = atoi(char_NUMDATA);

	if( NUMDATA <= 1000000 )
		strcpy( querySize,"1m");
	else if( NUMDATA == 2000000)
		strcpy( querySize,"2m");
	else 	if( NUMDATA == 4000000)
		strcpy( querySize,"4m");
	else 	if( NUMDATA == 8000000)
		strcpy( querySize,"8m");
	else
		strcpy( querySize,"16m");

	/*	else 	if( NUMDATA == 16000000 )
		strcpy( querySize,"16m");
	else 	if( NUMDATA == 32000000)
		strcpy( querySize,"32m");
	else 	if( NUMDATA == 40000000)
		strcpy( querySize,"40m");
	else
		strcpy( querySize,"1m");
		*/





	if( METHOD[7] == true)
		METHOD[0] = METHOD[1] = METHOD[2] =  METHOD[3] = METHOD[4] = METHOD[5] = METHOD[6] = true;


	if( (METHOD[0] || METHOD[1] || METHOD[2] ||  METHOD[3] || METHOD[4] || METHOD[5] || METHOD[6]) && NUMSEARCH == 0)
	{
		NUMSEARCH = 1000;
	}



	if(BUILD_TYPE == 0)
	{
		printf("DATATYPE : %s, PGSIZE %d, NUMDIMS %d, NUMBLOCK %d, NUMTHREADS %d,  NODECARD %d, NUMDATA %d, NUMSEARCH %d, SELECTION RATIO %s, NCPU %d,   PARTITIONED %d, ",
				    DATATYPE,      PGSIZE,    NUMDIMS,    NUMBLOCK,    NUMTHREADS,     NODECARD,    NUMDATA,    NUMSEARCH,    SELECTIVITY ,       NCPUCORES, PARTITIONED       );
	}
	else
	{
		printf("DATATYPE : %s, PGSIZE %d, NUMDIMS %d, NUMBLOCK %d, NUMTHREADS %d,  NODECARD %d, NUMDATA %d, NUMSEARCH %d, SELECTION RATIO %s, NCPU %d,   PARTITIONED %d, ",
	    		  DATATYPE,      BVH_PGSIZE,    NUMDIMS,    NUMBLOCK,    NUMTHREADS,     NODECARD,    NUMDATA,    NUMSEARCH,    SELECTIVITY ,       NCPUCORES, PARTITIONED       );
	}

	if( BUILD_TYPE == 0 )
		printf("\n\nRTrees will be build up.. \n");
	else if ( BUILD_TYPE == 1 || BUILD_TYPE == 2 )
		printf("\n\nBVH-Trees (TYPE : %d )  will be build up..\n", BUILD_TYPE);
	else
		printf("\n\nHilbertRadix Tree (TYPE : %d )  will be build up..\n", BUILD_TYPE);


	if( POLICY == 0 )
		printf("Original distribution\n");
	else
		printf("Roundrobin distribution\n");


	return true;
}



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
__device__ bool devRectOverlap(struct Rect *r, struct Rect *s)
{
	int i,j;

	for(i=0; i<NUMDIMS; i++){
		j = i + NUMDIMS;
		if(r->boundary[i] > s->boundary[j] || s->boundary[i] > r->boundary[j]){
			return false;
		}
	}
	return true;
}
__device__ bool dev_Node_SOA_Overlap(struct Rect *r, struct Node_SOA* n, int tid)
{
	int i,j, off1, off2;

	for(i=0; i<NUMDIMS; i++){
		j = i + NUMDIMS;
		off1 = i*NODECARD;
		off2 = j*NODECARD;

		if(r->boundary[i] > n->boundary[off2+tid] || n->boundary[off1+tid] > r->boundary[j]){
			return false;
		}
	}
	return true;
}
__device__ bool dev_BVH_Node_SOA_Overlap(struct Rect *r, BVH_Node_SOA* n, int tid)
{
	int i,j, off1, off2;


	for(i=0; i<NUMDIMS; i++){
		j = i + NUMDIMS;
		off1 = i*NODECARD;
		off2 = j*NODECARD;

		if(r->boundary[i] > n->boundary[off2+tid] || n->boundary[off1+tid] > r->boundary[j]){
			return false;
		}
	}
	return true;
}
bool RadixNode_SOA_Overlap(struct Rect *r, RadixTreeNode_SOA* n, int tid)
{
	int i, j, min, max;

	for(i=0; i<NUMDIMS; i++){
		j = i+NUMDIMS;
		min = NODECARD*(2*i+tid%2);
		max = NODECARD*(2*j+tid%2);

			//printf("tid %d r boundary[%d] %f > n boundary[%d] %f , n boundary[%d] %f > r boundary[%d] %f \n", 
			//		    tid, i, max+tid/2, min+tid/2, j,r->boundary[i], n->boundary[max+tid/2], n->boundary[min+tid/2], r->boundary[j]);
		if(r->boundary[i] > n->boundary[max+tid/2] || n->boundary[min+tid/2] > r->boundary[j]){
			return false;
		}
	}
	return true;
}

__device__ bool dev_RadixNode_SOA_Overlap(struct Rect *r, RadixTreeNode_SOA* n, int tid)
{
	int i,j, min, max;

	/*
	if( tid == 0 )
	{
		printf("r %f r %f r %f\n",r->boundary[0], r->boundary[1],r->boundary[2]);
		printf("r %f r %f r %f\n",r->boundary[3], r->boundary[4],r->boundary[5]);

		printf("n %f n %f n %f\n",n->boundary[0], n->boundary[NODECARD*2],n->boundary[NODECARD*4]);
		printf("n %f n %f n %f\n",n->boundary[NODECARD*6], n->boundary[NODECARD*8],n->boundary[NODECARD*10]);

		printf("n index %llu\n",n->index[0]);
	}
	__syncthreads();
	*/

	for(i=0; i<NUMDIMS; i++){
		j = i+NUMDIMS;
		min = NODECARD*(2*i+tid%2);
		max = NODECARD*(2*j+tid%2);

		if(r->boundary[i] > n->boundary[max+tid/2] || n->boundary[min+tid/2] > r->boundary[j]){
			return false;
		}
	}
	return true;
}

__device__ bool dev_Node_SOA_Overlap2(struct Rect *r, struct Node_SOA* n)
{
	int tid = threadIdx.x;
	int i,j, off1, off2;

	for(i=0; i<NUMDIMS; i++){
		j = i + NUMDIMS;
		off1 = i*NODECARD;
		off2 = j*NODECARD;

		if(r->boundary[i] > n->boundary[off2+tid] || n->boundary[off1+tid] > r->boundary[j]){
			return false;
		}
	}
	return true;
}
__device__ bool dev_BVH_Node_SOA_Overlap2(struct Rect *r, BVH_Node_SOA* n)
{
	int tid = threadIdx.x;
	int i,j, off1, off2;


	for(i=0; i<NUMDIMS; i++){
		j = i + NUMDIMS;
		off1 = i*NODECARD;
		off2 = j*NODECARD;

		if(r->boundary[i] > n->boundary[off2+tid] || n->boundary[off1+tid] > r->boundary[j]){
			return false;
		}
	}
	return true;
}

float IntersectedRectArea(struct Rect *r1, struct Rect *r2)
{
	int i,j;

	float area = 1.0f;

	for( i=0; i<NUMDIMS; i++)
	{
		j=i+NUMDIMS;
		area *= min( r1->boundary[j], r2->boundary[j])-max(r1->boundary[i], r2->boundary[i]);
	}
	return area;
}

void checkNodeOverlap(BVH_Node *n, float* area)
{
	if (n->level > 0) // this is an internal node in the tree //
	{
		for(int i1 = 0; i1<n->count-1; i1++)
			for(int i2 = i1+1; i2<n->count; i2++)
				if( RectOverlap(&n->branch[i1].rect, &n->branch[i2].rect) )
					*area += IntersectedRectArea(&n->branch[i1].rect, &n->branch[i2].rect);

		for (int i=0; i<n->count; i++)
			checkNodeOverlap(n->branch[i].child, area);
	}
}


int comp(const void * t1,const void * t2) 
{
	int* a = (int*)t1;
	int* b = (int*)t2;

	if (*a==*b)
		return 0;
	else
		if (*a < *b)
			return -1;
		else
			return 1;
}
int comp_d0(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[0] == b->rect.boundary[0]) // x axis
		return 0;
	else
		if ( a->rect.boundary[0] < b->rect.boundary[0]) // x axis
			return -1;
		else
			return 1;
}
int comp_d1(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[1] == b->rect.boundary[1]) // y axis
		return 0;
	else
		if ( a->rect.boundary[1] < b->rect.boundary[1]) // y axis
			return -1;
		else
			return 1;
}
int comp_d2(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[2] == b->rect.boundary[2]) // z axis
		return 0;
	else
		if ( a->rect.boundary[2] < b->rect.boundary[2]) // z axis
			return -1;
		else
			return 1;
}
int comp_d3(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[3] == b->rect.boundary[3]) // z axis
		return 0;
	else
		if ( a->rect.boundary[3] < b->rect.boundary[3]) // z axis
			return -1;
		else
			return 1;
}
#if NUMDIMS > 4
int comp_d4(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[4] == b->rect.boundary[4]) // z axis
		return 0;
	else
		if ( a->rect.boundary[4] < b->rect.boundary[4]) // z axis
			return -1;
		else
			return 1;
}int comp_d5(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[5] == b->rect.boundary[5]) // z axis
		return 0;
	else
		if ( a->rect.boundary[5] < b->rect.boundary[5]) // z axis
			return -1;
		else
			return 1;
}int comp_d6(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[6] == b->rect.boundary[6]) // z axis
		return 0;
	else
		if ( a->rect.boundary[6] < b->rect.boundary[6]) // z axis
			return -1;
		else
			return 1;
}int comp_d7(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[7] == b->rect.boundary[7]) // z axis
		return 0;
	else
		if ( a->rect.boundary[7] < b->rect.boundary[7]) // z axis
			return -1;
		else
			return 1;
}int comp_d8(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[8] == b->rect.boundary[8]) // z axis
		return 0;
	else
		if ( a->rect.boundary[8] < b->rect.boundary[8]) // z axis
			return -1;
		else
			return 1;
}int comp_d9(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[9] == b->rect.boundary[9]) // z axis
		return 0;
	else
		if ( a->rect.boundary[9] < b->rect.boundary[9]) // z axis
			return -1;
		else
			return 1;
}int comp_d10(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[10] == b->rect.boundary[10]) // z axis
		return 0;
	else
		if ( a->rect.boundary[10] < b->rect.boundary[10]) // z axis
			return -1;
		else
			return 1;
}int comp_d11(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[11] == b->rect.boundary[11]) // z axis
		return 0;
	else
		if ( a->rect.boundary[11] < b->rect.boundary[11]) // z axis
			return -1;
		else
			return 1;
}int comp_d12(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[12] == b->rect.boundary[12]) // z axis
		return 0;
	else
		if ( a->rect.boundary[12] < b->rect.boundary[12]) // z axis
			return -1;
		else
			return 1;
}int comp_d13(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[13] == b->rect.boundary[13]) // z axis
		return 0;
	else
		if ( a->rect.boundary[13] < b->rect.boundary[13]) // z axis
			return -1;
		else
			return 1;
}int comp_d14(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[14] == b->rect.boundary[14]) // z axis
		return 0;
	else
		if ( a->rect.boundary[14] < b->rect.boundary[14]) // z axis
			return -1;
		else
			return 1;
}int comp_d15(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[15] == b->rect.boundary[15]) // z axis
		return 0;
	else
		if ( a->rect.boundary[15] < b->rect.boundary[15]) // z axis
			return -1;
		else
			return 1;
}
#if NUMDIMS > 16
int comp_d16(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[16] == b->rect.boundary[16]) // z axis
		return 0;
	else
		if ( a->rect.boundary[16] < b->rect.boundary[16]) // z axis
			return -1;
		else
			return 1;
}int comp_d17(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[17] == b->rect.boundary[17]) // z axis
		return 0;
	else
		if ( a->rect.boundary[17] < b->rect.boundary[17]) // z axis
			return -1;
		else
			return 1;
}int comp_d18(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[18] == b->rect.boundary[18]) // z axis
		return 0;
	else
		if ( a->rect.boundary[18] < b->rect.boundary[18]) // z axis
			return -1;
		else
			return 1;
}int comp_d19(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[19] == b->rect.boundary[19]) // z axis
		return 0;
	else
		if ( a->rect.boundary[19] < b->rect.boundary[19]) // z axis
			return -1;
		else
			return 1;
}int comp_d20(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[20] == b->rect.boundary[20]) // z axis
		return 0;
	else
		if ( a->rect.boundary[20] < b->rect.boundary[20]) // z axis
			return -1;
		else
			return 1;
}
int comp_d21(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[21] == b->rect.boundary[21]) // z axis
		return 0;
	else
		if ( a->rect.boundary[21] < b->rect.boundary[21]) // z axis
			return -1;
		else
			return 1;
}int comp_d22(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[22] == b->rect.boundary[22]) // z axis
		return 0;
	else
		if ( a->rect.boundary[22] < b->rect.boundary[22]) // z axis
			return -1;
		else
			return 1;
}int comp_d23(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[23] == b->rect.boundary[23]) // z axis
		return 0;
	else
		if ( a->rect.boundary[23] < b->rect.boundary[23]) // z axis
			return -1;
		else
			return 1;
}int comp_d24(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[24] == b->rect.boundary[24]) // z axis
		return 0;
	else
		if ( a->rect.boundary[24] < b->rect.boundary[24]) // z axis
			return -1;
		else
			return 1;
}int comp_d25(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[25] == b->rect.boundary[25]) // z axis
		return 0;
	else
		if ( a->rect.boundary[25] < b->rect.boundary[25]) // z axis
			return -1;
		else
			return 1;
}int comp_d26(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[26] == b->rect.boundary[26]) // z axis
		return 0;
	else
		if ( a->rect.boundary[26] < b->rect.boundary[26]) // z axis
			return -1;
		else
			return 1;
}int comp_d27(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[27] == b->rect.boundary[27]) // z axis
		return 0;
	else
		if ( a->rect.boundary[27] < b->rect.boundary[27]) // z axis
			return -1;
		else
			return 1;
}int comp_d28(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[28] == b->rect.boundary[28]) // z axis
		return 0;
	else
		if ( a->rect.boundary[28] < b->rect.boundary[28]) // z axis
			return -1;
		else
			return 1;
}int comp_d29(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[29] == b->rect.boundary[29]) // z axis
		return 0;
	else
		if ( a->rect.boundary[29] < b->rect.boundary[29]) // z axis
			return -1;
		else
			return 1;
}int comp_d30(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[30] == b->rect.boundary[30]) // z axis
		return 0;
	else
		if ( a->rect.boundary[30] < b->rect.boundary[30]) // z axis
			return -1;
		else
			return 1;
}int comp_d31(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[31] == b->rect.boundary[31]) // z axis
		return 0;
	else
		if ( a->rect.boundary[31] < b->rect.boundary[31]) // z axis
			return -1;
		else
			return 1;
}int comp_d32(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[32] == b->rect.boundary[32]) // z axis
		return 0;
	else
		if ( a->rect.boundary[32] < b->rect.boundary[32]) // z axis
			return -1;
		else
			return 1;
}int comp_d33(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[33] == b->rect.boundary[33]) // z axis
		return 0;
	else
		if ( a->rect.boundary[33] < b->rect.boundary[33]) // z axis
			return -1;
		else
			return 1;
}int comp_d34(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[34] == b->rect.boundary[34]) // z axis
		return 0;
	else
		if ( a->rect.boundary[34] < b->rect.boundary[34]) // z axis
			return -1;
		else
			return 1;
}int comp_d35(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[35] == b->rect.boundary[35]) // z axis
		return 0;
	else
		if ( a->rect.boundary[35] < b->rect.boundary[35]) // z axis
			return -1;
		else
			return 1;
}int comp_d36(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[36] == b->rect.boundary[36]) // z axis
		return 0;
	else
		if ( a->rect.boundary[36] < b->rect.boundary[36]) // z axis
			return -1;
		else
			return 1;
}int comp_d37(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[37] == b->rect.boundary[37]) // z axis
		return 0;
	else
		if ( a->rect.boundary[37] < b->rect.boundary[37]) // z axis
			return -1;
		else
			return 1;
}int comp_d38(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[38] == b->rect.boundary[38]) // z axis
		return 0;
	else
		if ( a->rect.boundary[38] < b->rect.boundary[38]) // z axis
			return -1;
		else
			return 1;
}int comp_d39(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[39] == b->rect.boundary[39]) // z axis
		return 0;
	else
		if ( a->rect.boundary[39] < b->rect.boundary[39]) // z axis
			return -1;
		else
			return 1;
}int comp_d40(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[40] == b->rect.boundary[40]) // z axis
		return 0;
	else
		if ( a->rect.boundary[40] < b->rect.boundary[40]) // z axis
			return -1;
		else
			return 1;
}int comp_d41(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[41] == b->rect.boundary[41]) // z axis
		return 0;
	else
		if ( a->rect.boundary[41] < b->rect.boundary[41]) // z axis
			return -1;
		else
			return 1;
}int comp_d42(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[42] == b->rect.boundary[42]) // z axis
		return 0;
	else
		if ( a->rect.boundary[42] < b->rect.boundary[42]) // z axis
			return -1;
		else
			return 1;
}int comp_d43(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[43] == b->rect.boundary[43]) // z axis
		return 0;
	else
		if ( a->rect.boundary[43] < b->rect.boundary[43]) // z axis
			return -1;
		else
			return 1;
}int comp_d44(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[44] == b->rect.boundary[44]) // z axis
		return 0;
	else
		if ( a->rect.boundary[44] < b->rect.boundary[44]) // z axis
			return -1;
		else
			return 1;
}int comp_d45(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[45] == b->rect.boundary[45]) // z axis
		return 0;
	else
		if ( a->rect.boundary[45] < b->rect.boundary[45]) // z axis
			return -1;
		else
			return 1;
}int comp_d46(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[46] == b->rect.boundary[46]) // z axis
		return 0;
	else
		if ( a->rect.boundary[46] < b->rect.boundary[46]) // z axis
			return -1;
		else
			return 1;
}int comp_d47(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[47] == b->rect.boundary[47]) // z axis
		return 0;
	else
		if ( a->rect.boundary[47] < b->rect.boundary[47]) // z axis
			return -1;
		else
			return 1;
}int comp_d48(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[48] == b->rect.boundary[48]) // z axis
		return 0;
	else
		if ( a->rect.boundary[48] < b->rect.boundary[48]) // z axis
			return -1;
		else
			return 1;
}int comp_d49(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[49] == b->rect.boundary[49]) // z axis
		return 0;
	else
		if ( a->rect.boundary[49] < b->rect.boundary[49]) // z axis
			return -1;
		else
			return 1;
}int comp_d50(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[50] == b->rect.boundary[50]) // z axis
		return 0;
	else
		if ( a->rect.boundary[50] < b->rect.boundary[50]) // z axis
			return -1;
		else
			return 1;
}int comp_d51(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[51] == b->rect.boundary[51]) // z axis
		return 0;
	else
		if ( a->rect.boundary[51] < b->rect.boundary[51]) // z axis
			return -1;
		else
			return 1;
}int comp_d52(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[52] == b->rect.boundary[52]) // z axis
		return 0;
	else
		if ( a->rect.boundary[52] < b->rect.boundary[52]) // z axis
			return -1;
		else
			return 1;
}int comp_d53(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[53] == b->rect.boundary[53]) // z axis
		return 0;
	else
		if ( a->rect.boundary[53] < b->rect.boundary[53]) // z axis
			return -1;
		else
			return 1;
}
int comp_d54(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[54] == b->rect.boundary[54]) // z axis
		return 0;
	else
		if ( a->rect.boundary[54] < b->rect.boundary[54]) // z axis
			return -1;
		else
			return 1;
}int comp_d55(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[55] == b->rect.boundary[55]) // z axis
		return 0;
	else
		if ( a->rect.boundary[55] < b->rect.boundary[55]) // z axis
			return -1;
		else
			return 1;
}int comp_d56(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[56] == b->rect.boundary[56]) // z axis
		return 0;
	else
		if ( a->rect.boundary[56] < b->rect.boundary[56]) // z axis
			return -1;
		else
			return 1;
}int comp_d57(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[57] == b->rect.boundary[57]) // z axis
		return 0;
	else
		if ( a->rect.boundary[57] < b->rect.boundary[57]) // z axis
			return -1;
		else
			return 1;
}int comp_d58(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[58] == b->rect.boundary[58]) // z axis
		return 0;
	else
		if ( a->rect.boundary[58] < b->rect.boundary[58]) // z axis
			return -1;
		else
			return 1;
}int comp_d59(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[59] == b->rect.boundary[59]) // z axis
		return 0;
	else
		if ( a->rect.boundary[59] < b->rect.boundary[59]) // z axis
			return -1;
		else
			return 1;
}int comp_d60(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[60] == b->rect.boundary[60]) // z axis
		return 0;
	else
		if ( a->rect.boundary[60] < b->rect.boundary[60]) // z axis
			return -1;
		else
			return 1;
}int comp_d61(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[61] == b->rect.boundary[61]) // z axis
		return 0;
	else
		if ( a->rect.boundary[61] < b->rect.boundary[61]) // z axis
			return -1;
		else
			return 1;
}int comp_d62(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[62] == b->rect.boundary[62]) // z axis
		return 0;
	else
		if ( a->rect.boundary[62] < b->rect.boundary[62]) // z axis
			return -1;
		else
			return 1;
}int comp_d63(const void * t1,const void * t2) 
{
	BVH_Branch* a = (BVH_Branch*)t1;
	BVH_Branch* b = (BVH_Branch*)t2;

	if ( a->rect.boundary[63] == b->rect.boundary[63]) // z axis
		return 0;
	else
		if ( a->rect.boundary[63] < b->rect.boundary[63]) // z axis
			return -1;
		else
			return 1;
}
#endif
#endif


__global__ void globalSetDeviceRoot(char* buf, int partition_no, int NUMBLOCK, int PARTITIONED )
{ 

	if(partition_no==0){
		devNUMBLOCK = NUMBLOCK;
		deviceRoot = (struct Node**) malloc( sizeof(struct Node*) * PARTITIONED );
	}
	deviceRoot[partition_no] = (struct Node*) buf;
}
__global__ void globalSetDeviceBVHRoot(char* buf, int partition_no, int NUMBLOCK, int PARTITIONED )
{ 

	if(partition_no==0){
		devNUMBLOCK = NUMBLOCK;
		deviceBVHRoot = (BVH_Node**) malloc( sizeof(BVH_Node*) * PARTITIONED );
	}
	deviceBVHRoot[partition_no] = (BVH_Node*) buf;
}
__global__ void globalFreeDeviceRoot(int PARTITIONED )
{ 

	for(int i=0; i<PARTITIONED; i++)
		free(deviceRoot[i]);
//	free(deviceRoot);
}
__global__ void globalFreeDeviceBVHRoot(int PARTITIONED )
{ 
	for(int i=0; i<PARTITIONED; i++)
		free(deviceBVHRoot[i]);
//	free(deviceBVHRoot);
}


#if NUMDIMS < 64
__global__ void globaltranspose_node(int partition_no, int totalNodes)
{
	int tid = threadIdx.x;

	__shared__ struct Node_SOA node_soa;
	__shared__ char* node_ptr;
	__shared__ char* node_soa_ptr;

	node_ptr = (char*) deviceRoot[partition_no];
	node_soa_ptr = (char*)&node_soa;

	for( int i=0; i< totalNodes; i++ )
	{

		//memcpy a node to node_soa_ptr
		//transpose node -> node_SOA

		for( int d=0; d<NUMDIMS*2; d++)
			memcpy(node_soa_ptr+(tid*4)+(NODECARD*4)*d, node_ptr+8+(sizeof(struct Branch)*tid)+(d*4), sizeof(float)); 															//copy boundary
		memcpy(node_soa_ptr+(8*NODECARD*NUMDIMS)+tid*4, node_ptr+8+(sizeof(struct Branch)*tid)+sizeof(struct Rect), sizeof(int)); 								//copy index code
		memcpy(node_soa_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+tid*8, node_ptr+8+(sizeof(struct Branch)*tid)+sizeof(struct Rect)+8, sizeof(long)); //copy child pointer

		if( tid == 0 )
			memcpy(node_soa_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+(NODECARD*8), node_ptr, sizeof(int)*2); 																						//copy count and level
		__syncthreads();
		//and then copy to deviceRoot again
		//memcpy(node_ptr, node_soa_ptr, PGSIZE);	
		//writing global memory in parallel may cause a program
		for( int d=0; d<NUMDIMS*2; d++)
			memcpy(node_ptr+(tid*4)+(NODECARD*4)*d, node_soa_ptr+(tid*4)+(NODECARD*4)*d, sizeof(float));
		memcpy(node_ptr+(8*NODECARD*NUMDIMS)+tid*4, node_soa_ptr+(8*NODECARD*NUMDIMS)+tid*4, sizeof(int));
		memcpy(node_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+tid*8, node_soa_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+tid*8, sizeof(long));

		if( tid == 0 )
			memcpy(node_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+(NODECARD*8), node_soa_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+(NODECARD*8), sizeof(int)*2);
		__syncthreads();

		if( tid == 0)
			node_ptr += PGSIZE;
		__syncthreads();
	}
}

__global__ void globaltranspose_BVHnode(int partition_no, int totalNodes)
{
	int tid = threadIdx.x;

	__shared__ struct BVH_Node_SOA node_soa;
	__shared__ char* node_ptr;
	__shared__ char* node_soa_ptr;

	node_ptr = (char*) deviceBVHRoot[partition_no];
	node_soa_ptr = (char*)&node_soa;
	for( int i=0; i< totalNodes; i++ )
	{

		//memcpy a node to node_soa_ptr
		//transpose node -> node_SOA
		for( int d=0; d<NUMDIMS*2; d++)
			memcpy(node_soa_ptr+(tid*4)+(NODECARD*4)*d, node_ptr+8+(sizeof(BVH_Branch)*tid)+(d*4), sizeof(float)); 															//copy boundary
		memcpy(node_soa_ptr+(8*NODECARD*NUMDIMS)+tid*4, node_ptr+8+(sizeof(BVH_Branch)*tid)+sizeof(struct Rect), sizeof(int)); 								//copy index code
		memcpy(node_soa_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+tid*8, node_ptr+8+(sizeof(BVH_Branch)*tid)+sizeof(struct Rect)+8, sizeof(long)); //copy child pointer

		if( tid == 0 )
		{
			//copy count, level, parent and sibling pointer
			memcpy(node_soa_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+(NODECARD*8), node_ptr, sizeof(int)*2);
			memcpy(node_soa_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+(NODECARD*8)+8, node_ptr+8+sizeof(BVH_Branch)*NODECARD,sizeof(long)*2);
		}
		__syncthreads();

		//and then copy to deviceRoot again
		for( int d=0; d<NUMDIMS*2; d++)
			memcpy(node_ptr+(tid*4)+(NODECARD*4)*d, node_soa_ptr+(tid*4)+(NODECARD*4)*d, sizeof(float));
		memcpy(node_ptr+(8*NODECARD*NUMDIMS)+tid*4, node_soa_ptr+(8*NODECARD*NUMDIMS)+tid*4, sizeof(int));
		memcpy(node_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+tid*8, node_soa_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+tid*8, sizeof(long));

		if( tid == 0)
		memcpy(node_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+(NODECARD*8), node_soa_ptr+(8*NODECARD*NUMDIMS)+(NODECARD*4)+(NODECARD*8), (sizeof(int)*2)+(sizeof(long)*2));
		__syncthreads();

		if( tid == 0)
			node_ptr += BVH_PGSIZE;
		__syncthreads();

	}
}

#else

__global__ void globaltranspose_node(int partition_no, int totalNodes)
{
	struct Node_SOA node_soa;
	struct Node * node_ptr = (struct Node*) deviceRoot[partition_no];

	for( int i=0; i< totalNodes; i++ )
	{

		//memcpy a node to node_soa_ptr
		//transpose node -> node_SOA

		for(int b=0; b<NODECARD; b++)
		{
			for( int d=0; d<NUMDIMS*2; d++)
			{
				node_soa.boundary[d*NODECARD+b] = node_ptr->branch[b].rect.boundary[d];
			}

			node_soa.index[b] = (int) node_ptr->branch[b].hIndex;
			char* tmp = (char*) node_ptr;
			memcpy(&node_soa.child[b], tmp+8+(b*sizeof(struct Branch))+sizeof(struct Rect)+8, sizeof(long));
		}
		node_soa.count = node_ptr->count;
		node_soa.level = node_ptr->level;

		memcpy(node_ptr, &node_soa, PGSIZE);

		node_ptr++;
	}
}

__global__ void globaltranspose_BVHnode(int partition_no, int totalNodes)
{
	BVH_Node_SOA node_soa;
	BVH_Node * node_ptr = (BVH_Node*) deviceBVHRoot[partition_no];

	for( int i=0; i< totalNodes; i++ )
	{

		//memcpy a node to node_soa_ptr
		//transpose node -> node_SOA

		for(int b=0; b<NODECARD; b++)
		{
			for( int d=0; d<NUMDIMS*2; d++)
			{
				node_soa.boundary[d*NODECARD+b] = node_ptr->branch[b].rect.boundary[d];
			}

			node_soa.index[b] = (int) node_ptr->branch[b].mortonCode;
			char* tmp = (char*) node_ptr;
			memcpy(&node_soa.child[b], tmp+8+(b*sizeof(BVH_Branch))+sizeof(struct Rect)+8, sizeof(long));
		}
		node_soa.count = node_ptr->count;
		node_soa.level = node_ptr->level;

		node_soa.parent = (BVH_Node_SOA*) node_ptr->parent;
		node_soa.sibling = (BVH_Node_SOA*) node_ptr->sibling;

		memcpy(node_ptr, &node_soa, BVH_PGSIZE);

		node_ptr++;
	}
}

#endif


//	long nil_ptr = 0x0;
		/*
		if( tid == 0 )
		{
			struct Node* node = (struct Node*)node_ptr;
			printf("POINT1\n");

			for(int n=0; n<NODECARD; n++)
				memcpy(&node->branch[n].child, &nil_ptr ,sizeof(long) );

			for(int d=0; d<NUMDIMS*2; d++)
			{
				for(int n=0; n<node->count; n++)
					printf("%.6f \n", node->branch[n].rect.boundary[d]);
			}
			for(int n=0; n<node->count; n++)
				printf("%d \n", node->branch[n].hIndex);
			for(int n=0; n<node->count; n++)
			printf("%lu \n", node->branch[n].child);
		}
		__syncthreads();
		*/
		/*
		if( tid == 0)
		{
			printf("POINT2\n");
		for( int n=0; n<NODECARD*2*NUMDIMS; n++)
			printf("%.6f \n", node_soa.boundary[n]);

		for( int n=0; n<NODECARD; n++)
			printf("%d \n", node_soa.index[n]);

		for( int n=0; n<NODECARD; n++)
			printf("%lu \n", node_soa.child[n]);

		printf("%d %d\n", node_soa.count, node_soa.level);
		}
		*/


__global__ void globalDesignTraversalScenario()
{
	int tid = threadIdx.x;

	struct Node_SOA* root = (struct Node_SOA*) deviceRoot[0];
	struct Node_SOA* node = root;
	
	for( int i=0; i<129; i++)
	{
		for(  int d=0; d<NUMDIMS; d++)
		{
			node->boundary[tid+(d*NODECARD)] = 1.1f;
			node->boundary[tid+((d+NUMDIMS)*NODECARD)] = -0.1f;
		}
		
		if( i == 0 )// root node
		{
			if( tid == 0)
			{
				for(int d=0; d<NUMDIMS; d++)
				{
					node->boundary[2+(d*NODECARD)] = 0.0f;
					node->boundary[2+((d+NUMDIMS)*NODECARD)] = 1.0f;
					node->boundary[119+(d*NODECARD)] = 0.0f;
					node->boundary[119+((d+NUMDIMS)*NODECARD)] = 1.0f;
				}
			}
			__syncthreads();
		}
		else if( i == 3 )
		{
			if( tid == 0)
			{
				for(int d=0; d<NUMDIMS; d++)
				{
					node->boundary[(d*NODECARD)] = 0.0f;
					node->boundary[((d+NUMDIMS)*NODECARD)] = 1.0f;
				}
			}
			__syncthreads();
		}
		else if( i == 120 )
		{
			if( tid == 0)
			{
				for(int d=0; d<NUMDIMS; d++)
				{
					node->boundary[(d*NODECARD)] = 0.0f;
					node->boundary[((d+NUMDIMS)*NODECARD)] = 1.0f;
				}
			}
		}

		node++;
	}

	for( int i=0; i<16384; i++)
	{
		for(int d=0; d<NUMDIMS; d++)
		{
			node->boundary[tid+(d*NODECARD)] = 0.0f;
			node->boundary[tid+((d+NUMDIMS)*NODECARD)] = 1.0f;
		}
		node++;
	}
}
__global__ void globalDesignTraversalScenarioBVH()
{
	int tid = threadIdx.x;

	BVH_Node_SOA* root = (BVH_Node_SOA*) deviceBVHRoot[0];
	BVH_Node_SOA* node = root;
	
	for( int i=0; i<129; i++)
	{
		for(  int d=0; d<NUMDIMS; d++)
		{
			node->boundary[tid+(d*NODECARD)] = 1.1f;
			node->boundary[tid+((d+NUMDIMS)*NODECARD)] = -0.1f;
		}
		
		if( i == 0 )// root node
		{
			if( tid == 0)
			{
				for(int d=0; d<NUMDIMS; d++)
				{
					node->boundary[2+(d*NODECARD)] = 0.0f;
					node->boundary[2+((d+NUMDIMS)*NODECARD)] = 1.0f;
					node->boundary[119+(d*NODECARD)] = 0.0f;
					node->boundary[119+((d+NUMDIMS)*NODECARD)] = 1.0f;
				}
			}
			__syncthreads();
		}
		else if( i == 3 )
		{
			if( tid == 0)
			{
				for(int d=0; d<NUMDIMS; d++)
				{
					node->boundary[(d*NODECARD)] = 0.0f;
					node->boundary[((d+NUMDIMS)*NODECARD)] = 1.0f;
				}
			}
			__syncthreads();
		}
		else if( i == 120 )
		{
			if( tid == 0)
			{
				for(int d=0; d<NUMDIMS; d++)
				{
					node->boundary[(d*NODECARD)] = 0.0f;
					node->boundary[((d+NUMDIMS)*NODECARD)] = 1.0f;
				}
			}
		}

		node++;
	}

	for( int i=0; i<16384; i++)
	{
		for(int d=0; d<NUMDIMS; d++)
		{
			node->boundary[tid+(d*NODECARD)] = 0.0f;
			node->boundary[tid+((d+NUMDIMS)*NODECARD)] = 1.0f;
		}
		node++;
	}
}

int find_an_available_gpu(int num_of_gpus)
{
	int i;
	for(i=0; i<num_of_gpus; i++) 
	{
		cudaError_t error = cudaSetDevice(i);

		if( error != cudaSuccess)
			continue;

		size_t avail, total;
		cudaMemGetInfo( &avail, &total );
		size_t used = total-avail;
		float per = ( (double)used/(double)total)*100;

		if( per <= 10.0 )
		{
			printf("%dth GPU is selected \n", i);
	    return i;
		}
		else if( i == 7 )
		{
			char hostname[10];
			gethostname(hostname,10);
			printf("Unfortunately, There is no available device on %s",hostname);
			return 0;
		}
	}
	return -1;
}

//convert and print decimal in binary from most significant bit
void printDecimalToBinary(unsigned long long num, int order)
{
	if( order < 64 ) 
	{
	  printDecimalToBinary(num>>1, ++order);

	if( num % 2 == 0 )
		printf("0");
	else
		printf("1");
	}
}

//convert and print decimal in binary from least significant bit
// num : num to print out in binary
// order : bit position to print now 
// nbits : number of bits to print 
// valid bits : number of valid bits from most significant bit
void printCommonPrefix(unsigned long long num, int order, int nbits, int valid_bits)
{
	if( order < nbits ) 
	{
		printCommonPrefix(num>>1, ++order, nbits, valid_bits);

		if(order <= (64-valid_bits))
		{
			printf("X");
		}
		else
		{
			if( num % 2 == 0 )
				printf("0");
			else
				printf("1");
		}
	}
}

unsigned long long RightShift(unsigned long long val, int shift)
{
	  int loop = shift/30;
		  int i;

			  for(i=0; i < loop ; i++)
					    val = val >> 30;

				  val = val >> shift%30;

					  return val;
}
unsigned long long LeftShift(unsigned long long val, int shift)
{
	  int loop = shift/30;
		  int i;

			  for(i=0; i < loop ; i++)
					    val = val << 30;

				  val = val << shift%30;

					  return val;
}

void ConvertHilbertIndexToBoundingBox(unsigned long long index, int X ,float* rect)
{
	int i,j;
	int nbits = X%NUMDIMS;
	unsigned long long coord[NUMDIMS];
	unsigned long long coords[4][NUMDIMS];
	//unsigned long long coords[nbits*2][NUMDIMS];

	unsigned long long current_num;
	unsigned long long start_num = LeftShift( RightShift(index ,X), X);
	unsigned long long increments = LeftShift(1 , (X/NUMDIMS)*NUMDIMS) ;
	unsigned long long end_num =  start_num | (LeftShift(1 , X) -1);

//	printf("1 end num : ");
//	printCommonPrefix(end_num, 0 , 64, 64);
//	printf("\n\n");

	end_num = LeftShift(RightShift( end_num, (X/NUMDIMS)*NUMDIMS), (X/NUMDIMS)*NUMDIMS);

//	printf("2 end num : ");
//	printCommonPrefix(end_num, 0 , 64, 64);
//	printf("\n\n");

//	printf("X : %d\n", X);

//	printf("index : ");
//	printCommonPrefix(index, 0 , 64, 64);
//	printf("\n\n");

	for(i=0, current_num = start_num; current_num <= end_num; current_num+=increments, i++)
	{
		if( i!=0 && current_num <= start_num) break;

		/*
		if( index ==  61763094443385092)
		{
			printf("index : ");
			printCommonPrefix(index, 0 , 64, 64);
			printf("\n\n");

			printf("cur num : ");
			printCommonPrefix(current_num, 0 , 64, 64);
			printf("\n\n");

			printf("inc num : ");
			printCommonPrefix(increments, 0 , 64, 64);
			printf("\n\n");

			printf("end num : ");
			printCommonPrefix(end_num, 0 , 64, 64);
			printf("\n\n");
		}
		*/

//		printf("%d\n",__LINE__);
		hilbert_i2c(NUMDIMS, 20, current_num, coords[i]);
		for(j=0; j<NUMDIMS; j++)
		{
//		  printf("%d\n",__LINE__);
			coords[i][j] = LeftShift(RightShift(coords[i][j] , (X/NUMDIMS)) , (X/NUMDIMS));

			/*
			if( index ==  61763094443385092)
			{
				printf("coords[%d][%d] %llu\n", i,j,coords[i][j]);
			}
			*/
		}
	}

  hilbert_i2c(NUMDIMS, 20, index, coord);

	int diff_bits[NUMDIMS];
	for(i=0; i<NUMDIMS; i++)
	{
//		printf("%d\n",__LINE__);
		diff_bits[i]=0;
		if( nbits == 1 )
		diff_bits[i] = __builtin_popcountll( coords[0][i] ^ coords[1][i]);
		else if( nbits == 2 )
		diff_bits[i] = __builtin_popcountll( (coords[0][i]^coords[1][i])|(coords[0][i]^coords[2][i])|(coords[0][i]^coords[3][i]) );
		// TO DO :: nbits 3, 4 ~~

		diff_bits[i] += X/NUMDIMS;

		/*
		printf("org coord[%d] %llu \n", i, coord[i]);
		printCommonPrefix(coord[i] , 0 , 64, 64);
		printf("\n\n");
		*/

		coord[i] = LeftShift( RightShift( coord[i] , diff_bits[i]), diff_bits[i]);
		//printf("min coord[%d] %llu \n", i, coord[i]);
		//printCommonPrefix(coord[i] , 0 , 64, 64);
		//printf("\n");
		rect[i] = coord[i]/1000000.0f;
		//printf("min rect[%d] %f \n\n", i, rect[i]);

		coord[i] += (LeftShift( 1 , diff_bits[i] ) -1 );
		//printf("max coord[%d] %llu \n", i, coord[i]);
		//printCommonPrefix(coord[i] , 0 , 64, 64);
		//printf("\n");
		rect[i+NUMDIMS] = coord[i]/1000000.0f;
		//printf("max rect[%d] %f \n\n", i+NUMDIMS, rect[i+NUMDIMS]);

	}

}

