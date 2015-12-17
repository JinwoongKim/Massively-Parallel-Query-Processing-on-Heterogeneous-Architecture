#if !defined(_host_h_)
#define _host_h_
#include <index.h>
//#####################################################################
//###################### SEARCH FUNCTIONS #############################
//#####################################################################

void searchDataOnCPU();
int hostLinearSearchData(struct Branch* b, struct Rect* r );
int hostRTreeSearch(struct Node *n, struct Rect *r , int *vNode);
int hostBVHTreeSearch(struct Node *n, struct Rect *r , int *vNode);
void* PThreadSearch(void* arg);
void* PThreadSearch_BVH(void* arg);
void RTree_Multicore();
void BVHTree_Multicore();

#endif
