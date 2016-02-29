#include "node/g_node.h"

#include <cassert>

namespace ursus {
namespace node {

NodeType G_Node::GetNodeType(void) const {
  return node_type;
}
int G_Node::GetLevel(void) const {
  return level;
}

void G_Node::SetPoint(ui offset, Point point) {
  assert(offset < GetNumberOfDims()*2*GetNumberOfDegrees());
  points[offset] = point;
}

void G_Node::SetIndex(ui offset, ull _index) {
  assert(offset < GetNumberOfDegrees());
  index[offset] = _index;
}

void G_Node::SetChild(ui offset, G_Node* _child) {
  assert(offset < GetNumberOfDegrees());
  child[offset] = _child;
}

void G_Node::SetNodeType(NodeType type) {
  assert(type);
  node_type = type;
}

void G_Node::SetLevel(int _level) {
  level = _level;
}

void G_Node::SetBranchCount(ui _branch_count) {
  assert(_branch_count < GetNumberOfDegrees());
  branch_count = _branch_count;
}

} // End of node namespace
} // End of ursus namespace
