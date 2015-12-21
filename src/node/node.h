#pragma once

#include "common/types.h"

#include <vector>

namespace ursus {
namespace node {

class Branch;

typedef class Node* Node_Ptr;

class Node{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//

 //===--------------------------------------------------------------------===//
 // Accessor
 //===--------------------------------------------------------------------===//

 Branch GetBranch(ui offset) const;
 ui GetBranchCount(void) const;
 NodeType GetNodeType(void) const;
 ui GetLevel(void) const;

 void SetNodeType(NodeType type);
 void SetLevel(ui level);

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
