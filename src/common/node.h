#pragma once

#include "common/branch.h"

namespace ursus {

//FIXME load below variables from somewhere ...
extern number_of_childs;

class Node {
  private:
    //TODO change to unsigned int ?
    int count;
    //TODO change to unsigned int ?
    int level; /* 0 is leaf, others positive */
    Branch branch[number_of_childs]; 
};

class Node_SOA
{
	float boundary[NODECARD*2*NUMDIMS]; // 8*NODECARD*NUMDIMS
	int index[NODECARD]; // 4*NODECARD, hilbert curve index or morton code
	struct Node_SOA* child[NODECARD]; // 8*NODECARD
	int count; // int(4byte)
	int level; // int(4byte) /* 0 is leaf, others positive */
	int dummy[NODECARD];
};



} // End of ursus namespace
