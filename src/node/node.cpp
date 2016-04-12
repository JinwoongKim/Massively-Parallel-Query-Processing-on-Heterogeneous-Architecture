#include "node/node.h"

#include "common/macro.h"
#include "node/branch.h"

#include <cassert>

namespace ursus {
namespace node {

//===--------------------------------------------------------------------===//
// Accessor
//===--------------------------------------------------------------------===//
__both__
Branch Node::GetBranch(ui offset) const {
  assert(offset < branch_count);
  return branches[offset];
}

__both__
ui Node::GetBranchCount(void) const {
  return branch_count;
}

__both__
Point Node::GetBranchPoint(ui branch_offset, ui point_offset) const{
  return branches[branch_offset].GetPoint(point_offset);
}

__both__
ll Node::GetBranchIndex(ui branch_offset) const{
  return branches[branch_offset].GetIndex();
}

__both__
ll Node::GetLastBranchIndex(void) const{
  return branches[branch_count-1].GetIndex();
}

__both__
ll Node::GetBranchChildOffset(ui branch_offset) const{
  return branches[branch_offset].GetChildOffset();
}

__both__
NodeType Node::GetNodeType(void) const {
  return node_type;
}

__both__
int Node::GetLevel(void) const {
  return level;
}

__both__
void Node::SetBranch(Branch _branch, ui offset) {
  branches[offset++] = _branch;
  branch_count = (offset>branch_count)?offset:branch_count;
}

__both__
void Node::SetBranchCount(ui _branch_count) {
  branch_count = _branch_count;
}

__both__
void Node::SetBranchPoint(Point point, 
                          ui branch_offset, ui point_offset) {
  branches[branch_offset].SetPoint(point, point_offset);
}

__both__
void Node::SetBranchIndex(ll index, ui branch_offset) {
  branches[branch_offset].SetIndex(index);
}

__both__
void Node::SetBranchChildOffset(ui branch_offset, ll child_offset) {
  branches[branch_offset].SetChildOffset(child_offset);
}

__both__
void Node::SetNodeType(NodeType type) {
  assert(type);
  node_type = type;
}

__both__
void Node::SetLevel(int _level) {
  level = _level;
}

bool Node::IsOverlap(Point* query, ui branch_offset) {

  for(ui range(lower_boundary, 0, GetNumberOfDims())) {
    int upper_boundary = lower_boundary+GetNumberOfDims();  

    if (query[lower_boundary] > branches[branch_offset].GetPoint(upper_boundary) ||
        query[upper_boundary] < branches[branch_offset].GetPoint(lower_boundary)) {
      return false;
    }
  }

  return true;
}

// Get a string representation
std::ostream &operator<<(std::ostream &os, const Node &node) {
  os << " Node : " << std::endl;
  os << " NodeType = " << NodeTypeToString(node.GetNodeType()) << std::endl;
  os << " NodeLevel = " << node.GetLevel() << std::endl;
  os << " Branch Count = " << node.GetBranchCount() << std::endl;
  for( ui range(i, 0, node.GetBranchCount())) {
    os << " Branch["<< i << "] : " << node.GetBranch(i) << std::endl;
  }
  return os;
}

} // End of node namespace
} // End of ursus namespace
