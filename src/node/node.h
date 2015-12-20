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
  // branches
  std::vector<Branch> branches;

  NodeType node_type = NODE_TYPE_INVALID;

  // distance from root
  unsigned int level;
};

} // End of node namespace
} // End of ursus namespace
