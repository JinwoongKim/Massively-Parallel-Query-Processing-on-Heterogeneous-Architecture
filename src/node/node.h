#pragma once

#include "common/types.h"
#include "node/branch.h"

namespace ursus {
namespace node {

typedef class Node* Node_Ptr;

class Node{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  // branches
  std::vector<Branch> branches;

  // node type
  NodeType node_type = NODE_TYPE_INVALID;

  // distance from root
  ui level;
};

} // End of node namespace
} // End of ursus namespace
