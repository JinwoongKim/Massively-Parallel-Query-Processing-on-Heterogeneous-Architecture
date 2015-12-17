#if !defined(_mpts_h_)
#define _mpts_h_

#include <index.h>

//#####################################################################
//###################### SEARCH FUNCTIONS #############################
//#####################################################################


void MPTS(int number_of_procs, int myrank);
__global__ void globalMPTS(struct Rect* query, int* hit, int* count, int* rootCount,  int mpiSEARCH,  int PARTITIONED );
__device__ long devFindLeftmost(int partition_index , struct Rect *r, int* count );
__device__ long devFindRightmost(int partition_index, struct Rect *r, int* count );

__global__ void globalMPTS_ILP(struct Rect* query, int* hit, int* count, int* rootCount,  int mpiSEARCH,  int PARTITIONED );
__device__ long devFindLeftmost_ILP(int partition_index, struct Rect *r, int* count );
__device__ long devFindRightmost_ILP(int partition_index, struct Rect *r,int* count );

#endif
