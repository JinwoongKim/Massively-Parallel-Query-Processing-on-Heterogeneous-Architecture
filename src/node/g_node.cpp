#include "node/g_node.h"

#include <assert>

namespace ursus {
namespace node {

NodeType G_Node::GetNodeType(void) const {
  return node_type;
}
ui G_Node::GetLevel(void) const {
  return level;
}

void G_Node::SetPoint(ui offset, Point _point) {
  assert(offset < GetNumberOfDims()*2*GetNumberOFDegrees());
  point[offset] = _point;
}

void G_Node::SetIndex(ui offset, ull _index) {
  assert(offset < GetNumberOFDegrees());
  index[offset] = _index;
}

void G_Node::SetChild(ui offset, G_Node_Ptr _child) {
  assert(offset < GetNumberOFDegrees());
  child[offset] = _child;
}

void G_Node::SetNodeType(NodeType type) {
  assert(type);
  node_type = type;
}

void G_Node::SetLevel(ui _level) {
  level = level;
}

void G_Node::SetBranchCount(ui _branch_count) {
  assert(_branch_count < GetNumberOFDegrees());
  branch_count = _branch__count;
}

} // End of node namespace
} // End of ursus namespace
