#pragma once

#include "common/config.h"
#include "common/types.h"

namespace ursus {
namespace node {

class G_Node{
 public:
 //===--------------------------------------------------------------------===//
 // Accessor
 //===--------------------------------------------------------------------===//

 NodeType GetNodeType(void) const;
 int GetLevel(void) const;

 void SetPoint(ui offset, Point point);
 void SetIndex(ui offset, ull index);
 void SetChild(ui offset, G_Node* child);
 void SetNodeType(NodeType type);
 void SetLevel(int level);
 void SetBranchCount(ui branch_count);

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  // transformed branches
  Point points[GetNumberOfDims()*2*GetNumberOfDegrees()];
  ull index[GetNumberOfDegrees()];
  G_Node* child[GetNumberOfDegrees()];

  // node type
  NodeType node_type = NODE_TYPE_INVALID;

  // distance from root
  int level;

  // branch_count
  ui branch_count;
};

} // End of node namespace
} // End of ursus namespace
