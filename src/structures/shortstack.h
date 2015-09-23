#if !defined(_shortstack_h_)
#define _shortstack_h_

#include <index.h>


#if NUMDIMS == 3
#if NODECARD == 2
  #define STACK_SIZE 25
#elif NODECARD == 4
  #define STACK_SIZE 15
#elif NODECARD == 8
  #define STACK_SIZE 9
#elif NODECARD == 16
  #define STACK_SIZE 7
#elif NODECARD == 32
  #define STACK_SIZE 5
#elif NODECARD == 64
  #define STACK_SIZE 5
#elif NODECARD == 128
  #define STACK_SIZE 4
#elif NODECARD == 256
  #define STACK_SIZE 4
#elif NODECARD == 512
  #define STACK_SIZE 2
#endif
#else
#if NODECARD == 128
  #if NUMDIMS == 2
  	#define STACK_SIZE 11
  #elif NUMDIMS == 4
  	#define STACK_SIZE 7
  #elif NUMDIMS == 8
  	#define STACK_SIZE 4
  #elif NUMDIMS == 16
  	#define STACK_SIZE 2
  #elif NUMDIMS == 32
  	#define STACK_SIZE 1
	#else
  	#define STACK_SIZE 0
  #endif
#elif NODECARD == 256
  #if NUMDIMS == 2
  	#define STACK_SIZE 5
  #elif NUMDIMS == 4
  	#define STACK_SIZE 3
  #elif NUMDIMS == 8
  	#define STACK_SIZE 2
  #elif NUMDIMS == 16
  	#define STACK_SIZE 1
	#else
  	#define STACK_SIZE 4
  #endif
#else 
  #if NUMDIMS == 2
  	#define STACK_SIZE 23
  #elif NUMDIMS == 4
  	#define STACK_SIZE 15
  #elif NUMDIMS == 8
  	#define STACK_SIZE 9
  #elif NUMDIMS == 16
  	#define STACK_SIZE 5
  #elif NUMDIMS == 32
  	#define STACK_SIZE 2
  #elif NUMDIMS == 64
  	#define STACK_SIZE 1
	#else
  	#define STACK_SIZE 4
  #endif
#endif
#endif

//#####################################################################
//###################### SEARCH FUNCTIONS #############################
//#####################################################################


void shortstack(int number_of_procs, int myrank);
__global__ void globalShortstack(struct Rect* query, int * hit, int* count , int* rootCount, int* pushCount, int* popCount,  int mpiSEARCH, int PARTITIONED  );
__global__ void globalShortstack_BVH (struct Rect* query, int * hit, int* count , int* rootCount, int* pushCount, int* popCount, int mpiSEARCH, int PARTITIONED);

__global__ void globalShortstack_ILP(struct Rect* query, int * hit, int mpiSEARCH, int PARTITIONED  );
__global__ void globalShortstack_ILP_BVH(struct Rect* query, int * hit, int mpiSEARCH, int PARTITIONED  );

__global__ void globalShortstack_TP(struct Rect* query, int * hit, int* count , int* rootCount, int* pushCount, int* popCount,  int mpiSEARCH, int PARTITIONED  );
//__global__ void globalShortstack_BVH_TP (struct Rect* query, int * hit, int* count , int* rootCount, int* pushCount, int* popCount, int mpiSEARCH, int PARTITIONED);

#endif
