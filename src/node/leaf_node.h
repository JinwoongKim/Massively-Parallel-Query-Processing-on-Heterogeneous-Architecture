#pragma once

#include "common/config.h"
#include "common/types.h"

#include "node/branch.h"

namespace ursus {
namespace node {

class LeafNode {
 public:
 //===--------------------------------------------------------------------===//
 // Accessor
 //===--------------------------------------------------------------------===//

 __both__  std::vector<Point> GetMBB() const;
 __both__  Branch GetBranch(ui offset) const;
 __both__  ui GetBranchCount(void) const;
 __both__ Point GetBranchPoint(ui branch_offset, ui point_offset) const;
 __both__ ll GetBranchIndex(ui branch_offset) const;
 __both__ ll GetLastBranchIndex(void) const;
 __both__ ll GetBranchChildOffset(ui branch_offset) const;
 __both__ LeafNode* GetBranchChildLeafNode(ui branch_offset) const;
 __both__ NodeType GetNodeType(void) const;
 __both__ int GetLevel(void) const;

 __both__ void SetBranch(Branch branch, ui offset);
 __both__ void SetBranchCount(ui branch_count);
 __both__ void SetBranchPoint(ui branch_offset, Point point, ui point_offset);
 __both__ void SetBranchIndex(ui branch_offset, ll index);
 __both__ void SetBranchChildOffset(ui branch_offset, ll child_offset);
 __both__ void SetNodeType(NodeType type);
 __both__ void SetLevel(int level);

 bool IsOverlap(Point* query, ui branch_offset);
 bool IsOverlap(ui branch_offset, ui branch_offset2);

  // Get a string representation for debugging
  friend std::ostream &operator<<(std::ostream &os, const LeafNode &node);
 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  // branches
  Branch branches[GetNumberOfLeafNodeDegrees()];

  // # of branch
  ui branch_count=0;

  // node type
  NodeType node_type = NODE_TYPE_INVALID;

  // distance from root
  int level=-1;
};

} // End of node namespace
} // End of ursus namespace
