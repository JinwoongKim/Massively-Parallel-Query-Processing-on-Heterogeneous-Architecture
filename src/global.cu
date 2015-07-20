#include "global.h"

//##########################################################################
//########################### GLOBAL VARIABLES #############################
//##########################################################################

int number_of_procs, myrank;
pthread_barrier_t bar;

int NUMDATA, NUMBLOCK, NUMSEARCH, PARTITIONED, WORKLOAD, NCPUCORES, POLICY, BUILD_TYPE, DEVICE_ID;
int keepDoing;
bool METHOD[8]; // MPES, MPTS, MPHR, MPHR2, ShortStack, ParentLink, SkipPointer

float t_time[7], t_visit[7], t_rootVisit[7], t_pop[7], t_push[7], t_parent[7], t_skipPointer[7];

char querySize[20], SELECTIVITY[20];

//roots for cpu
char** ch_root;
char** cd_root; // store leaf nodes temporarily

unsigned long long indexSize[2];
int boundary_of_trees, tree_height[2], *number_of_node_in_level[2];
int numOfleafNodes;

class BVH_Node;
class BVH_Branch;

__device__ long **parent_offset;
__device__ struct Node** deviceRoot;
__device__ class BVH_Node** deviceBVHRoot;
__device__ struct RadixTreeNode_SOA** deviceRadixRoot;
__device__ int devNUMBLOCK;

float elapsed_time;

// queues
queue<struct split_range*> sq;
queue<BVH_Node*> bvh_q;
queue<struct RadixTreeNode*> radix_q0;
queue<struct RadixTreeNode*> radix_q1;
queue<struct RadixTreeNode*> radix_q2;

priority_queue<struct RadixTreeNode*, vector<struct RadixTreeNode*>, class RadixTreeNodeComparator> radix_q3;

//bool RadixTreeNodeComp(const RadixTreeNode& a, const RadixTreeNode& b);

//priority_queue<RadixTreeNode*, vector<RadixTreeNode*>, RadixTreeNodeComparator> radix_q2;


priority_queue<BVH_Node*, vector<BVH_Node*>, NodeComparator> p_nodeQueue;


