#if !defined(_mpes_h_)
#define _mpes_h_

#include <index.h>
//#####################################################################
//########################### MPES ####################################
//#####################################################################


void MPES(int number_of_procs, int myrank);
__global__ void globalMPES(struct Rect * query, int *b_hit, long * d_leafNode_offset, int* d_numberOfLeafnodes,  int mpiSEARCH, int boundary_index, int PARTITIONED);
__global__ void globalMPES_ILP(struct Rect * query, int *b_hit, long * d_leafNode_offset, int* d_numberOfLeafnodes,  int mpiSEARCH, int boundary_index, int PARTITIONED);
__global__ void globalMPES_BVH(struct Rect * query, int *b_hit, int mpiSEARCH,  int PARTITIONED);
__global__ void globalMPES_ILP_BVH(struct Rect * query, int *b_hit, int mpiSEARCH,  int PARTITIONED );
__global__ void globalMPES_TP(struct Rect * query, int *b_hit, long * d_leafNode_offset, int* d_numberOfLeafnodes,  int mpiSEARCH, int boundary_index, int PARTITIONED);
__global__ void globalMPES_BVH_TP(struct Rect * query, int *b_hit, int mpiSEARCH,  int PARTITIONED);
__global__ void globalMPES_RadixArray(struct Rect * query, int *b_hit, int mpiSEARCH,  int PARTITIONED);

#endif
