#pragma once

#include "common/config.h"
#include "common/types.h"

#include <iostream>

namespace ursus {
namespace node {

class Node_SOA{
 public:
 //===--------------------------------------------------------------------===//
 // Accessor
 //===--------------------------------------------------------------------===//

 __both__ NodeType GetNodeType(void) const;
 __both__ int GetLevel(void) const;
 __both__ ui GetBranchCount(void) const;
 __both__ ll GetIndex(ui offset) const;
 __both__ ll GetLastIndex() const;
 __both__ ll GetChildOffset(ui offset) const;
 __both__ Node_SOA* GetChildNode(ui offset) const;
  Point GetPoint(ui offset) const;
  Point GetBranchPoint(ui branch_offset, ui dim) const;

 void SetPoint(ui offset, Point point);
 void SetBranchPoint(ui branch_offset, Point point, ui dim);
 void SetIndex(ui offset, ll index);
 void SetChildOffset(ui offset, ll child_offset);
 void SetNodeType(NodeType type);
 void SetLevel(int level);
 void SetBranchCount(ui branch_count);

 __both__  bool IsOverlap(Point* query, ui child_offset);

 friend std::ostream &operator<<(std::ostream &os, const Node_SOA &node_soa);
 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  // transformed branches
  Point points[GetNumberOfDims()*2*GetNumberOfLeafNodeDegrees()];
  ll index[GetNumberOfLeafNodeDegrees()];
  ll child_offset[GetNumberOfLeafNodeDegrees()];

  // node type
  NodeType node_type = NODE_TYPE_INVALID;

  // distance from root
  int level;

  // branch_count
  ui branch_count;
};

} // End of node namespace
} // End of ursus namespace
