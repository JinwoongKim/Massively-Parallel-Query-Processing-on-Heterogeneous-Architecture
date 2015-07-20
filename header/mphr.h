#if !defined(_mphr_h_)
#define _mphr_h_

#include <index.h>

//#####################################################################
//###################### SEARCH FUNCTIONS #############################
//#####################################################################


void MPHR(int number_of_procs, int myrank);
__global__ void globalMPHR(struct Rect* query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED  );
__global__ void globalMPHR_ILP(struct Rect* query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED  );
__global__ void globalMPHR_BVH(struct Rect* query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED  );
__global__ void globalMPHR_ILP_BVH(struct Rect* query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED  );
__global__ void globalMPHR_RadixTree (struct Rect* _query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED);

void MPHR2(int number_of_procs, int myrank);

__global__ void globalMPHR2(struct Rect* query, int * hit, int* count , long* _leafNode_offset, long* _extendLeafNode_offset , int* rootCount, int mpiSEARCH, int boundary_of_trees, int PARTITIONED  );
__global__ void globalMPHR2_BVH(struct Rect* query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED  );

__global__ void globalMPHR2_TP(struct Rect* query, int * hit, int* count , long* _leafNode_offset, long* _extendLeafNode_offset , int* rootCount, int mpiSEARCH, int boundary_of_trees, int PARTITIONED  );
__global__ void globalMPHR2_BVH_TP(struct Rect* query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED  );

__global__ void globalMPHR2_ILP(struct Rect* query, int * hit, int* count , long* _leafNode_offset, long* _extendLeafNode_offset , int* rootCount, int mpiSEARCH, int boundary_of_trees, int PARTITIONED  );
__global__ void globalMPHR2_ILP_BVH(struct Rect* query, int * hit, int* count , int* rootCount, int mpiSEARCH, int PARTITIONED  );

#endif

