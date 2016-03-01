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

 NodeType GetNodeType(void) const;
 int GetLevel(void) const;

 void SetPoint(ui offset, Point point);
 void SetIndex(ui offset, ull index);
 void SetChild(ui offset, Node_SOA* child);
 void SetNodeType(NodeType type);
 void SetLevel(int level);
 void SetBranchCount(ui branch_count);

 friend std::ostream &operator<<(std::ostream &os, const Node_SOA &node_soa);
 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  // transformed branches
  Point points[GetNumberOfDims()*2*GetNumberOfDegrees()];
  ull index[GetNumberOfDegrees()];
  Node_SOA* child[GetNumberOfDegrees()];

  // node type
  NodeType node_type = NODE_TYPE_INVALID;

  // distance from root
  int level;

  // branch_count
  ui branch_count;
};

} // End of node namespace
} // End of ursus namespace
