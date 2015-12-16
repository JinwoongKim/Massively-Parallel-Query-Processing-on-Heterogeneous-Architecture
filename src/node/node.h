#pragma once

#include "common/types.h"
#include "node/branch.h"

namespace ursus {
namespace node {

class Node{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  NodeType node_type = NODE_TYPE_INVALID;

  // distance from root
  unsigned int level;

  // branches
  std::vector<Branch> branches;
};

} // End of node namespace
} // End of ursus namespace
