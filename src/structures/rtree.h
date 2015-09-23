
#if !defined(_rtree_h_ )
#define _rtree_h_

#include <index.h>

//#####################################################################
//################## INSERTION and SORT FUNCTIONS #####################
//#####################################################################

void InsertDataRTrees(bitmask_t* keys, struct Branch* data);
int HilbertCompare(const void* a, const void* b);
__global__ void globalReassignhilbertIndex(struct Branch* data, int NUMDATA );

bool TreeDumpToFile();
bool TreeLoadFromFile();
void TreeLoadFromMem(struct Node* n, char *buf);
__global__ void globalLoadFromMem(char* d_treeRoot, int partition_no, int tNODE);
__global__ void globalTreeLoadFromMem(char *buf , int partitioned_number, int NUMBLOCK, int PARTITIONED, int numberOfnodes);
__device__ void devTreeLoadFromMem(struct Node* n, char *buf, int numberOfnodes);

//#####################################################################
//######################## BUILD FUNCTIONS ############################
//#####################################################################

void Build_Rtrees(int m);
void Bulk_loading( struct Branch* data );
void Bulk_loading_with_parentLink( struct Branch* data );
void Bulk_loading_with_skippointer( struct Branch* data );
__global__ void globalSetDeviceRoot(char* buf, int partition_no, int NUMBLOCK, int PARTITIONED );
__global__ void globalSetDeviceBVHRoot(char* buf, int partition_no, int NUMBLOCK, int PARTITIONED );
__global__ void globalBottomUpBuild(long *_offset, long *_parent_offset, int *_number_of_node, int boundary_index, int PARTITIONED); 
__global__ void globalBottomUpBuild_ILP(long *_offset, long *_parent_offset, int *_number_of_node, int boundary_index, int PARTITIONED);
__global__ void globalBottomUpBuild_with_parentLink(long *_offset, long *_parent_offset, int *_number_of_node, int boundary_index, int PARTITIONED); 
__global__ void globalBottomUpBuild_ILP_with_parentLink(long *_offset, long *_parent_offset, int *_number_of_node, int boundary_index, int PARTITIONED);

//FOR skip pointer
void checkDumpedTree(char* buf, int totalNodes);
unsigned long long BVHTreeDumpToMemDFS(char* buf, char* buf2,  unsigned long long childOff, unsigned long long* off );
void LinkUpSibling2( char* buf,  int totalNodes);
__global__ void globalBVH_Skippointer(int partition_no, int tree_height, int totalNodes);
void BVHTreeDumpToMem(BVH_Node* n, char *buf, int tree_height); // Breadth-First Search
__global__ void globalBVHTreeLoadFromMem(char *buf, int partition_no, int NUMBLOCK, int PARTITIONED /* number of partitioned index*/, int numOfnodes);


__global__ void global_print_BVH(int partition_no, int tNODE);

#endif
