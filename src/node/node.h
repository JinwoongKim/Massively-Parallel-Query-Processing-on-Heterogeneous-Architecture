#pragma once

#include "common/types.h"

namespace ursus {
namespace node {

class Node{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  Node(){}

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  NodeType node_type = NODE_TYPE_INVALID;

  // distance from root
  unsigned int level;

  // FIXME MBB
  //???

  // child pointers 
  Node* child;
  
};

} // End of node namespace
} // End of ursus namespace
