#if !defined(_skippointer_h_)
#define _skippointer_h_

#include <index.h>

//#####################################################################
//###################### SEARCH FUNCTIONS #############################
//#####################################################################


void skippointer(int number_of_procs, int myrank);
__global__ void globalSkippointer_BVH(struct Rect* query, int * hit, int* count , int* rootCount, int* skipCount, int mpiSEARCH, int PARTITIONED  );
__global__ void globalSkippointer_BVH_TP(struct Rect* query, int * hit, int* count , int* rootCount, int* skipCount, int mpiSEARCH, int PARTITIONED  );
__global__ void globalSkippointer_ILP_BVH(struct Rect* query, int * hit, int mpiSEARCH, int PARTITIONED  );

#endif
