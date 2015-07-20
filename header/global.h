#if !defined(__global_h__)
#define __global_h__

#include <queue>  // using queue C++ style
#include <iostream>
#include <index.h>
using namespace std;

//##########################################################################
//########################### GLOBAL VARIABLES #############################
//##########################################################################

//MPI
extern int number_of_procs, myrank;

extern pthread_barrier_t bar;

//Arguments
extern int NUMDATA, NUMBLOCK, NUMSEARCH, PARTITIONED, WORKLOAD, NCPUCORES, POLICY, BUILD_TYPE, DEVICE_ID;
extern int keepDoing;
extern bool METHOD[8]; // MPES, MPTS, MPHR, MPHR2, ShortStack, ParentLink, SkipPointer

extern float t_time[7], t_visit[7], t_rootVisit[7], t_pop[7], t_push[7], t_parent[7], t_skipPointer[7];

extern char querySize[20], SELECTIVITY[20];

//roots for cpu
extern char** ch_root;
extern char** cd_root; // store leaf nodes temporarily

extern unsigned long long indexSize[2];
extern int boundary_of_trees, tree_height[2], *number_of_node_in_level[2];
extern int numOfleafNodes;

class BVH_Node;
class BVH_Branch;

extern __device__ long **parent_offset;
extern __device__ struct Node** deviceRoot;
extern __device__ class BVH_Node** deviceBVHRoot;
extern __device__ struct RadixTreeNode_SOA** deviceRadixRoot;

extern __device__ int devNUMBLOCK;


extern float elapsed_time;

// queues
extern queue<struct split_range*> sq;
extern queue<BVH_Node*> bvh_q;
extern queue<struct RadixTreeNode*> radix_q0;
extern queue<struct RadixTreeNode*> radix_q1;
extern queue<struct RadixTreeNode*> radix_q2;

struct RadixTreeNodeComparator;
struct NodeComparator;

extern priority_queue<struct RadixTreeNode*, vector<struct RadixTreeNode*>, RadixTreeNodeComparator> radix_q3;

//bool RadixTreeNodeComp(const RadixTreeNode& a, const RadixTreeNode& b);

//priority_queue<RadixTreeNode*, vector<RadixTreeNode*>, RadixTreeNodeComparator> radix_q2;


extern priority_queue<BVH_Node*, vector<BVH_Node*>, NodeComparator> p_nodeQueue;

#endif
