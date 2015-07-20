
#if !defined(__common_h__)
#define __common_h__

#include <stdio.h>
#include <index.h>
#include <structures.h>
#include <global.h>
//for cuda checking
#include <assert.h>

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

void profileCopies(float        *h_a, float        *h_b, float        *d, unsigned int  n, char         *desc);
void printHelp(char **argv);
bool ParseArgs(int argc, char **argv) ;
bool  RectOverlap(struct Rect *r, struct Rect *s);
__device__ bool devRectOverlap(struct Rect *r, struct Rect *s);
__device__ bool dev_Node_SOA_Overlap(struct Rect *r, struct Node_SOA* n, int tid);
__device__ bool dev_BVH_Node_SOA_Overlap(struct Rect *r, BVH_Node_SOA* n, int tid);
bool RadixNode_SOA_Overlap(struct Rect *r, RadixTreeNode_SOA* n, int tid);
__device__ bool dev_RadixNode_SOA_Overlap(struct Rect *r, RadixTreeNode_SOA* n, int tid);
__device__ bool dev_Node_SOA_Overlap2(struct Rect *r, struct Node_SOA* n);
__device__ bool dev_BVH_Node_SOA_Overlap2(struct Rect *r, BVH_Node_SOA* n);
float IntersectedRectArea(struct Rect *r1, struct Rect *r2);
void checkNodeOverlap(BVH_Node *n, float* area);

int comp(const void * t1,const void * t2) ;
int comp_d0(const void * t1,const void * t2) ;
int comp_d1(const void * t1,const void * t2) ;
int comp_d2(const void * t1,const void * t2) ;
int comp_d3(const void * t1,const void * t2) ;
#if NUMDIMS > 4
int comp_d4(const void * t1,const void * t2) ;
int comp_d5(const void * t1,const void * t2) ;
int comp_d6(const void * t1,const void * t2) ;
int comp_d7(const void * t1,const void * t2) ;
int comp_d8(const void * t1,const void * t2) ;
int comp_d9(const void * t1,const void * t2) ;
int comp_d10(const void * t1,const void * t2) ;
int comp_d11(const void * t1,const void * t2) ;
int comp_d12(const void * t1,const void * t2) ;
int comp_d13(const void * t1,const void * t2) ;
int comp_d14(const void * t1,const void * t2) ;
int comp_d15(const void * t1,const void * t2) ;
#if NUMDIMS > 16
int comp_d16(const void * t1,const void * t2) ;
int comp_d17(const void * t1,const void * t2) ;
int comp_d18(const void * t1,const void * t2) ;
int comp_d19(const void * t1,const void * t2) ;
int comp_d20(const void * t1,const void * t2) ;
int comp_d21(const void * t1,const void * t2) ;
int comp_d22(const void * t1,const void * t2) ;
int comp_d23(const void * t1,const void * t2) ;
int comp_d24(const void * t1,const void * t2) ;
int comp_d25(const void * t1,const void * t2) ;
int comp_d26(const void * t1,const void * t2) ;
int comp_d27(const void * t1,const void * t2) ;
int comp_d28(const void * t1,const void * t2) ;
int comp_d29(const void * t1,const void * t2) ;
int comp_d30(const void * t1,const void * t2) ;
int comp_d31(const void * t1,const void * t2) ;
int comp_d32(const void * t1,const void * t2) ;
int comp_d33(const void * t1,const void * t2) ;
int comp_d34(const void * t1,const void * t2) ;
int comp_d35(const void * t1,const void * t2) ;
int comp_d36(const void * t1,const void * t2) ;
int comp_d37(const void * t1,const void * t2) ;
int comp_d38(const void * t1,const void * t2) ;
int comp_d39(const void * t1,const void * t2) ;
int comp_d40(const void * t1,const void * t2) ;
int comp_d41(const void * t1,const void * t2) ;
int comp_d42(const void * t1,const void * t2) ;
int comp_d43(const void * t1,const void * t2) ;
int comp_d44(const void * t1,const void * t2) ;
int comp_d45(const void * t1,const void * t2) ;
int comp_d46(const void * t1,const void * t2) ;
int comp_d47(const void * t1,const void * t2) ;
int comp_d48(const void * t1,const void * t2) ;
int comp_d49(const void * t1,const void * t2) ;
int comp_d50(const void * t1,const void * t2) ;
int comp_d51(const void * t1,const void * t2) ;
int comp_d52(const void * t1,const void * t2) ;
int comp_d53(const void * t1,const void * t2) ;
int comp_d54(const void * t1,const void * t2) ;
int comp_d55(const void * t1,const void * t2) ;
int comp_d56(const void * t1,const void * t2) ;
int comp_d57(const void * t1,const void * t2) ;
int comp_d58(const void * t1,const void * t2) ;
int comp_d59(const void * t1,const void * t2) ;
int comp_d60(const void * t1,const void * t2) ;
int comp_d61(const void * t1,const void * t2) ;
int comp_d62(const void * t1,const void * t2) ;
int comp_d63(const void * t1,const void * t2) ;
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
