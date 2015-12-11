
#if !defined(__common_h__)
#define __common_h__

#include <stdio.h>
#include <index.h>
#include <structures.h>
#include <global.h>
//for cuda checking
#include <assert.h>

#define range(i,a,b) i = (a); i < (b); ++i

inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n",
      cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void printHelp(char **argv);
bool ParseArgs(int argc, char **argv) ;
__device__ bool dev_Node_SOA_Overlap(struct Rect *r, struct Node_SOA* n, int tid);
__device__ bool dev_BVH_Node_SOA_Overlap(struct Rect *r, BVH_Node_SOA* n, int tid);
bool RadixNode_SOA_Overlap(struct Rect *r, RadixTreeNode_SOA* n, int tid);
__device__ bool dev_RadixNode_SOA_Overlap(struct Rect *r, RadixTreeNode_SOA* n, int tid);
__device__ bool dev_Node_SOA_Overlap2(struct Rect *r, struct Node_SOA* n);
__device__ bool dev_BVH_Node_SOA_Overlap2(struct Rect *r, BVH_Node_SOA* n);
void checkNodeOverlap(BVH_Node *n, float* area);

int comp(const void * t1,const void * t2) ;
int comp_dim(const void * t1,const void * t2) ;

#endif
#endif
__global__ void globalSetDeviceRoot(char* buf, int partition_no, int NUMBLOCK, int PARTITIONED );
__global__ void globalSetDeviceBVHRoot(char* buf, int partition_no, int NUMBLOCK, int PARTITIONED );
__global__ void globalFreeDeviceRoot(int PARTITIONED );
__global__ void globalFreeDeviceBVHRoot(int PARTITIONED );



#if NUMDIMS < 64
__global__ void globaltranspose_node(int partition_no, int totalNodes);
__global__ void globaltranspose_BVHnode(int partition_no, int totalNodes);
#else
__global__ void globaltranspose_node(int partition_no, int totalNodes);
__global__ void globaltranspose_BVHnode(int partition_no, int totalNodes);
#endif
__global__ void globalDesignTraversalScenario();
__global__ void globalDesignTraversalScenarioBVH();
int find_an_available_gpu(int num_of_gpus);
//convert and print decimal in binary from most significant bit
void printDecimalToBinary(unsigned long long num, int order);
//convert and print decimal in binary from least significant bit
// num : num to print out in binary
// order : bit position to print now 
// nbits : number of bits to print 
// valid bits : number of valid bits from most significant bit
void printCommonPrefix(unsigned long long num, int order, int nbits, int valid_bits);
unsigned long long RightShift(unsigned long long val, int shift);
unsigned long long LeftShift(unsigned long long val, int shift);
void ConvertHilbertIndexToBoundingBox(unsigned long long index, int X ,float* rect);
#endif
