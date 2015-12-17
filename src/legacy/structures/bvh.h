
#if !defined(_bvh_h_)
#define _bvh_h_
#define DEBUG 0

#include <index.h>

//#####################################################################
//################## INSERTION and SORT FUNCTIONS #####################
//#####################################################################
//
void InsertDataBVH(bitmask_t* keys, BVH_Branch* data);
int MortonCompare(const void* a, const void* b);

//#####################################################################
//######################## BUILD FUNCTIONS ############################
//#####################################################################

void Build_BVH(int index_type);
int findSplit( BVH_Branch *data, int first,int  last);
void findSplits(BVH_Branch* data, int first, int last, int* split_pos, int *split_pos_cnt);
BVH_Node* generateHierarchy( BVH_Branch* data, int first, int last, int level, unsigned long long  *totalNodes, int *tree_height );
void BVHTreeDumpToMem(BVH_Node* n, char *buf, int tree_height); // Breadth-First Search
unsigned long long BVHTreeDumpToMemDFS(char* buf, char* buf2,  unsigned long long childOff, unsigned long long* off );
void LinkUpSibling( char* buf, int tree_height, int totalNodes);
void LinkUpSibling2( char* buf,  int totalNodes);
void settingBVH_trees(BVH_Node* n, int *numOfnodes, const int reverse );
void settingBVH_trees2(BVH_Node* n, int *numOfnodes, const int reverse, int totalNodes );
void traverseBVH_BFS(BVH_Node* n); // Breadth-First Search
void checkDumpedTree(char* buf, int totalNodes);

__global__ void globalBVHTreeLoadFromMem(char *buf, int partition_no, int NUMBLOCK, int PARTITIONED /* number of partitioned index*/, int numOfnodes);
__device__ void devBVHTreeLoadFromMem(BVH_Node* n, char *buf, int numOfnodes);

//void BVH_RecursiveSearch();
__device__ int devBVHTreeSearch(BVH_Node *n, struct Rect *r);
//__global__ void globalBVHTreeSearchRecursive(struct Rect* query, int* hit, int mpiSEARCH, int PARTITIONED);
void print_BVHNode(BVH_Node* n );
void print_BVHBranch(BVH_Branch* n );
__device__ void device_print_BVHnode(BVH_Node* n );
__global__ void global_print_BVH(int partition_no, int tNODE);
__global__ void globalBVH_Skippointer(int partition_no, int tree_height, int totalNodes);

#endif
