#pragma once

namespace ursus {

#include <stdio.h>
#include <iostream>
#include "index.h"
//using namespace std;

//##########################################################################
//############################ STRUCTURES ##################################
//##########################################################################

struct Node;
class BVH_Node;

struct Node_SOA;
struct Node_SOA
{
	float boundary[NODECARD*2*NUMDIMS]; // 8*NODECARD*NUMDIMS
	int index[NODECARD]; // 4*NODECARD, hilbert curve index or morton code
	struct Node_SOA* child[NODECARD]; // 8*NODECARD
	int count; // int(4byte)
	int level; // int(4byte) /* 0 is leaf, others positive */
	int dummy[NODECARD];
};

struct  BVH_Node_SOA;
struct  BVH_Node_SOA
{
	float boundary[NODECARD*2*NUMDIMS]; // 8*NODECARD*NUMDIMS
	int index[NODECARD]; // 4*NODECARD
	struct BVH_Node_SOA* child[NODECARD]; // 8*NODECARD
	int count; // int(4byte)
	int level; // int(4byte)
	struct BVH_Node_SOA* sibling; // 8
	struct BVH_Node_SOA* parent; // 8
	int dummy[NODECARD];
};

/****************************** HilbertRadixTree ******************************/

struct  RadixTreeNode_SOA;
struct  RadixTreeNode
{
	struct RadixTreeNode* child[2]; // 8*NODECARD
	unsigned long long common_prefix; // common prefix for children
	unsigned long long index[2]; // the largest hilbert index of its left and right sub tree 
	int nbits; // bit significance
	int level; // the level of a current node
	RadixTreeNode_SOA* child_soa;

};

struct  RadixTreeNode_SOA
{
	struct RadixTreeNode_SOA* child[NODECARD*2]; // 8*NODECARD
	unsigned long long index[NODECARD*2]; // hilbert index for children
	unsigned long long common_prefix[NODECARD]; // common prefix for children
	float boundary[NODECARD*4*NUMDIMS];
	int nbits[NODECARD];
	int level[NODECARD]; // the level of a current node
	int count; // number of childs 
};

/*
struct  _RadixTreeLeafNodes
{
	unsigned long long index[NUMDATA];
	unsigned long long common_prefix[NUMDATA]; 
	float boundary[NUMDATA*2*NUMDIMS];
};
typedef struct  _RadixTreeLeafNodes RadixTreeLeafNodes;
*/



//struct  RadixTreeNode_SOA;
//struct  RadixTreeNode_SOA
//{
//	struct RadixTreeNode_SOA* child[NODECARD]; // 8*NODECARD
//	unsigned long long common_prefix[NODECARD/2]; // common prefix for children
//	int nbits[NODECARD/2];
//	int index[NODECARD/2]; // hilbert index for children
//	int level[NODECARD/2]; // the level of a current node
//	int count; // number of childs 
//	float boundary[NODECARD*2*NUMDIMS];
//};

struct RadixTreeNodeComparator
{
	bool operator() (RadixTreeNode* arg1, RadixTreeNode* arg2)
	{
		if( arg1->index[0] < arg2->index[0] )
		{
			return false;
		}
		else if( arg1->index[0] == arg2->index[0] )
		{
			if( arg1 < arg2 )
				return false;
			else
				return true;
		}
		else
		{
			return true;
		}
	}
};

/*************************************************************************/


struct thread_data{
	int  tid;
	int  nblock;
	long  hit;
	int vNode;
	struct Node* root;
};

struct thread_data_BVH{
	int  tid;
	int  nblock;
	long  hit;
	int vNode;
	BVH_Node* root;
};




class BVH_Branch  //  16 + 8*NUMDIMS (byte)
{
public:
	struct Rect rect;		// 8 + 8*NUMDIMS (byte)
	unsigned long long mortonCode;
	// unsigned long long (8byte)
	BVH_Node *child; // Pointer (8byte)
public:
	/*
	BVH_Branch& operator=(const BVH_Branch &b)
	{
		for(int i =0 ; i <NUMDIMS; i++)
		{
			rect.boundary[i] = b.rect.boundary[i];
		}
		mortonCode = b.mortonCode;
		return *this;
	}
	*/
	void printBranch()
	{
		for(int i =0 ; i <NUMDIMS; i++)
		{
			printf("%.6f ",rect.boundary[i]);
		}
		printf("\n code %lu\n",mortonCode);
		printf("child %lu\n",child);
	}

	void setRectBoundary(int i, float boundary)
	{
		rect.boundary[i] = boundary;
	}
	void setMortonCode(unsigned long long code)
	{
		mortonCode = code;
	}
	void setChild(BVH_Node* c)
	{
		child = c;
	}
	float getRectBoundary(int i)
	{
		return rect.boundary[i];
	}
	unsigned long long getMortonCode()
	{
		return mortonCode;
	}
	BVH_Node*  getChild()
	{
		return child;
	}
};

struct split_range
{
	int leftbound;
	int rightbound;
};

struct NodeComparator
{
	bool operator() (BVH_Node* arg1, BVH_Node* arg2)
	{
		if( arg1->getLevel() < arg2->getLevel() )
			return true;
		else{
			if( arg1->getBranchMortonCode(0) > arg2->getBranchMortonCode(0))
				return true;
		}
		return false;
	}
};


} // End of ursus namespace
