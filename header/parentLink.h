#if !defined(_parent_h_)
#define _parent_h_

#include <index.h>

//#####################################################################
//###################### SEARCH FUNCTIONS #############################
//#####################################################################


void ParentLink(int number_of_procs, int myrank);
//__global__ void globalParentLink_BVH(struct Rect* query, int * hit, int* count , int* rootCount, int* fromChildCount, int* fromSiblingCount, int* fromParentCount, int mpiSEARCH, int PARTITIONED  );
__global__ void globalParentLink_BVH(struct Rect* query, int * hit, int* count , int* rootCount, int* parentCount,  int mpiSEARCH, int PARTITIONED  );
__global__ void globalParentLink_ILP_BVH(struct Rect* query, int * hit, int mpiSEARCH, int PARTITIONED  );

__global__ void globalParentLink_BVH_TP(struct Rect* query, int * hit, int* count , int* rootCount, int* parentCount,  int mpiSEARCH, int PARTITIONED  );

#endif
