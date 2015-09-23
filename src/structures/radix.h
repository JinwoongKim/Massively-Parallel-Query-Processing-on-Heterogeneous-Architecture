// Tree Traversal :: 1. store ractangle and traverse tree structure using MPRS
//                   2. calculate ractangle and traverse tree structure using MPRS
//                   3. shrink bounding box when traversing tree structure from root to leaf 
//                   4. traverse tree structure via comparing common prefix and given range query's hilbert index 
//
//
//
// Tree Structure :: 5. Auxiliary leaf nodes can be two different types, one is RadixTreeNode_SOA and another is RadixTreeLeafNode
//                      (without auxiliary leaf nodes, node utilization is almost 3 % )
//                   6. Leaf nodes are redundant(should be deleted from the memory) because we have RadixTreeNode_SOA or RadixTreeLeafNode
//                   9. Leaf node utilization is low... 
//
//                   8. Leaf nodes only use left child, so that thread can access data in sparsely 
//                   7. When the degree of tree is 128, child pointers can be 128 or 256,
//
//
// TO DO :: 1. To improve experiments, tree index can be dumped into the file
//
// Problems :: 1. It shows a different output when we convert multi-dimensional data into hilbert index, and vice versa. 
//             2. Some of data has the same hilbert index..
//             3. Internal nodes have bounding boxes that aren't minimum 
//             4. Link phase should be fast
//
//
// MAIN FOCUS :: 1. Compare the number of visited tree nodes between HilbertRadixTree and MPHR-tree
//               2. How much of space can be reduced by HilbertRadixTree
//


#if !defined(_radix_h_)
#define _radix_h_

#include <index.h>


#define LEAF_NODE INT_MAX
#define AUXILIARY_LEAF_NODE 0

//#####################################################################
//#################### CONSTRUCTION FUNCTIONS #########################
//#####################################################################

void InsertData(bitmask_t* data, float* data_rect); // for hilbert radix tree
void Build_RadixTrees(int index_type);
int findSplitAndCommonPrefix( bitmask_t *data, int first,int  last, int *nbits);
RadixTreeNode* generateRadixTreesHierarchy( bitmask_t* data, RadixTreeNode_SOA* RadixTreeNodeArray, int first, int last, int level, unsigned long long*  numOfNode );
int CopyDataToLeafNodes(RadixTreeNode_SOA* soa_array, bitmask_t* data, float* data_rect );
int mappingTreeIntoArray(RadixTreeNode* root, RadixTreeNode_SOA* soa_array, bitmask_t* data, float* data_rect, unsigned long long* start_leaf_nodes, unsigned long long _array_index );
void RadixArrayDumpToMem(RadixTreeNode_SOA* root, char* buf, int numOfnodes);
__global__ void globalRadixTreeLoadFromMem(char *buf, int partition_no, int NUMBLOCK, int PARTITIONED /* number of partitioned index*/, int numOfnodes);
__device__ void devRadixTreeLoadFromMem(RadixTreeNode_SOA* root, char *buf, int numOfnodes);
__global__ void globalPrintRadixTree(int PARTITIONED, int numOfNode, int , int, int*);
__device__ void devicePrintRadixTree(RadixTreeNode_SOA* node);
void settingRadixTrees(RadixTreeNode* n, const int reverse);

void traverseRadixTrees_BFS(RadixTreeNode* n); 
//int SettingIndexInRadixTree(RadixTreeNode_SOA* n);
void PrintRadixTree(RadixTreeNode_SOA* n);
int TraverseRadixTree(RadixTreeNode_SOA* n, struct Rect *r );
void ScanningHilbertRadixTree(RadixTreeNode_SOA* RadixTreeNodeArray, unsigned long long start_leaf_nodes, unsigned long long numOfNodes);
void TraverseHilbertRadixTreeUsingMPHR(RadixTreeNode_SOA* root);
void TraverseHilbertRadixTreeUsingMPHR2(RadixTreeNode_SOA* root);
void TraverseHilbertRadixTreeUsingStack(RadixTreeNode_SOA* root);


#endif
