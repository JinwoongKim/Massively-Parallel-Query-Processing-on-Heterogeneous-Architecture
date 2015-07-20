#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include "hilbert.h"

 //#include <malloc.h>
 //#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "sys/time.h"
 //#include <sys/types.h>
 //#include <sys/stat.h>
 #include <cuda.h>
 //#include <fcntl.h>
 //#include <unistd.h>
//#include <pthread.h>
//#include <queue>	// using queue C++ style
 //#include <getopt.h>
//#include "index.h"
//#include "split_q.h"


//Thrust library
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

using namespace std;

#define NUMDIMS	3		/* number of dimensions */
#define NUMBLOCK 128	/* number of block */
#define NUMTHREADS 256	/* number of nodecard */
#define NODECARD 256	/* number of nodecard */
#define PGSIZE	(int) ( (sizeof(int)*2) + ( NODECARD*sizeof(struct Branch) ) )


//##########################################################################
//########################### GLOBAL VARIABLES #############################
//##########################################################################



//FOR DEBUG
int indexSize;
//arguments
int NUMDATA, SEARCH;
float sRatio;
char dataType[5], querySize[5];

int hitsum = 0;
int sum = 0;

//host root of index
char* ch_root;

//device root of index
__device__ struct Node* deviceRoot;


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

void hostLinearSearchData(struct Branch* b, struct Rect* r ,int * distArray, int * hitArray)
{
	//FILE *fp = fopen("pattern", "w+");

	int hit = 0;
	int pp = -1; //previous position

	//fseek(fp, 0 , SEEK_END);

	for( int i = 0 ; i < NUMDATA; i+= NODECARD) 
	{
		int p = i/NODECARD  ; //position
		//		printf("position check %d\n",position);


		for( int j = 0; j < NODECARD && i+j < NUMDATA; j ++  ){
			if( RectOverlap(&b[i+j].rect, r) ){
				hit++;
				hitsum++;
			}
		}

		/*
			 printf("hIndex %d || ",b[i].hIndex);
			 for( int d = 0 ; d < 2*NUMDIMS; d++)
			 printf("%5.5f ", b[i].rect.boundary[d]);
			 printf("\n");
		 */

		if ( hit > 0 )
		{
			if ( pp == - 1 )
			{
				pp = p;
			}


			//if( p-pp > 1 )
			//{
				hitArray[hit]++;
				hit = 0;
			//}

			distArray[p-pp]++;
			sum++;

			//fprintf(fp,"hit %d dist %d \n",hit, p-pp);

			pp = p;
		}
	}

	
	//fprintf(fp,"\n\n");

}




void searchDataOnCPU(struct Branch* b, int *data, int*hit)
{
	char queryFileName[100];

	if(sRatio == 0)
		sprintf(queryFileName, "/home/jwkim/inputFile/%s_dim_query.%d.bin.1s.%s",dataType, NUMDIMS, querySize );
	else if(sRatio == 0.5)
		sprintf(queryFileName, "/home/jwkim/inputFile/%s_dim_query.%d.bin.halfs.%s", dataType,NUMDIMS,querySize);
	else 
		sprintf(queryFileName, "/home/jwkim/inputFile/%s_dim_query.%d.bin.%ds.%s", dataType,NUMDIMS,(int)sRatio,querySize);

	FILE *fp = fopen(queryFileName, "r");
	if(fp==0x0){
		printf("Line %d : query file open error\n",__LINE__);
		exit(1);
	}



	//host search
	struct Rect r;

	for( int i = 0 ; i < SEARCH; i++){
		for(int j=0;j<2*NUMDIMS;j++){
			fread(&r.boundary[j], sizeof(float), 1, fp);
//			printf("query %d %5.5f\n",j,r.boundary[j]);
		}
		hostLinearSearchData(b, &r, data, hit );
	}

	fclose(fp);
	
}





void initARGS(int argc, char* args[]){

	NUMDATA = atoi(args[optind]);
	if(argc > 2)
		SEARCH = atoi(args[optind+1]);
	strcpy( dataType , "real" );

	if(argc > 3)
		sRatio = atof(args[optind+2]);

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



	printf("PGSIZE %d NUMDIMS %d, NODECARD %d, NUMDATA %d, SEARCH %d ", PGSIZE, NUMDIMS, NODECARD, NUMDATA, SEARCH );
	if( argc > 3 )
		printf("sRatio %.2f ", sRatio );
	//if( argc > 4 ) 
	//	printf("dataType is %s", dataType);

	cout << endl;
}

void insertData(int* keys, struct Branch* data){

	char dataFileName[100];

	//sprintf(dataFileName, "/home/jwkim/inputFile/%s_dim_data.%d.bin\0", dataType, NUMDIMS); 
	sprintf(dataFileName, "/home/jwkim/inputFile/NOAA.bin");

	FILE *fp = fopen(dataFileName, "r");
	if(fp==0x0){
		printf("Line %d : Insert file open error\n", __LINE__);
		printf("%s\n", dataFileName);
		exit(1);
	}

	//bitmask_t nBits = ((int)(log(NUMDATA)/log(2))+1);
	//!!! nBits*NUMDIMS can not equal and larger than 64(bitmask_t)
	// but nBits should be larger than 1.
	//int nDims = min ( 63, NUMDIMS ) ;

	struct Branch b;
	for(int i=0; i<NUMDATA; i++){

		bitmask_t coord[NUMDIMS];
		for(int j=0;j<NUMDIMS;j++){
			fread(&b.rect.boundary[j], sizeof(float), 1, fp);
			b.rect.boundary[NUMDIMS+j] = b.rect.boundary[j];

			coord[j] = (unsigned)(1000000*b.rect.boundary[j]);
		}

		b.hIndex = hilbert_c2i(NUMDIMS, 20,coord);
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

	return (unsigned)(b1->hIndex-b2->hIndex);
}


int main(int argc, char *args[])
{

	//Initilizing with arguments
	initARGS(argc, args);

	//####################################################
	//################# INSERT DATA ######################
	//####################################################

	//insert data and calculate the Hilbert value 
	int keys[NUMDATA];
	struct Branch data[NUMDATA];
	insertData(keys, data);


	//####################################################
	//################## SORT ON CPU  ####################
	//####################################################
	qsort(data, NUMDATA, sizeof(struct Branch), HilbertCompare);


	int distArray[NUMDATA];
	int hitArray[NUMDATA];
	for( int i = 0 ; i < NUMDATA; i++) 
	{
	 	distArray[i] =  0  ;
		hitArray[i] = 0 ; 
	}

	searchDataOnCPU(data,distArray,hitArray);


	for( int i = 0 ; i < NUMDATA; i++) 
		if( hitArray[i] != 0 )
			printf("hit %d count %d percentage %5.2f(%) \n",i, hitArray[i], (hitArray[i]/(float)hitsum) * 100 );

	printf("hitsum %d \n",hitsum );

	for( int i = 0 ; i < NUMDATA; i++) 
		if( distArray[i] != 0)
			printf("dist %d count %d percentage %5.2f(%) \n",i, distArray[i], (distArray[i]/(float)sum) * 100 );


	printf("dist sum %d \n",sum );





	cout << "END" << endl;
	return 0;
}

