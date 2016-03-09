#pragma once

#include "common/config.h"
#include "common/types.h"

#include "node/branch.h"

namespace ursus {
namespace node {

class Node {
 public:
 //===--------------------------------------------------------------------===//
 // Accessor
 //===--------------------------------------------------------------------===//

 __both__  Branch GetBranch(ui offset) const;
 __both__  ui GetBranchCount(void) const;
 __both__ Point GetBranchPoint(ui branch_offset, ui point_offset) const;
 __both__ ull GetBranchIndex(ui branch_offset) const;
 __both__ ull GetLastBranchIndex(void) const;
 __both__ ull GetBranchChildOffset(ui branch_offset) const;
 __both__ NodeType GetNodeType(void) const;
 __both__ int GetLevel(void) const;

 __both__ void SetBranch(Branch branch, ui offset);
 __both__ void SetBranchCount(ui branch_count);
 __both__ void SetBranchPoint(Point point, ui branch_offset, ui point_offset);
 __both__ void SetBranchIndex(ull index, ui branch_offset);
 __both__ void SetBranchChildOffset(ui branch_offset, ull child_offset);
 __both__ void SetNodeType(NodeType type);
 __both__ void SetLevel(int level);

 bool IsOverlap(Point* query, ui branch_offset);

  // Get a string representation for debugging
  friend std::ostream &operator<<(std::ostream &os, const Node &node);
 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  // branches
  Branch branches[GetNumberOfDegrees()];

  // # of branch
  ui branch_count=0;

  // node type
  NodeType node_type = NODE_TYPE_INVALID;

  // distance from root
  int level=-1;
};

} // End of node namespace
} // End of ursus namespace
