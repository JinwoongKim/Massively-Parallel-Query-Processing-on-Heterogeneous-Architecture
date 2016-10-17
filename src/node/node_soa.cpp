#include "common/macro.h"
#include "node/node_soa.h"

#include <cassert>
#include <iomanip>

namespace ursus {
namespace node {

__both__
NodeType Node_SOA::GetNodeType(void) const {
  return node_type;
}

__both__
int Node_SOA::GetLevel(void) const {
  return level;
}

__both__
ui Node_SOA::GetBranchCount(void) const {
  return branch_count;
}

__both__
ll Node_SOA::GetIndex(ui offset) const {
  assert(offset < branch_count);
  return index[offset];
}

__both__
ll Node_SOA::GetLastIndex(void) const {
  return index[branch_count-1];
}

__both__
ll Node_SOA::GetChildOffset(ui offset) const {
  assert(offset < branch_count);
  return child_offset[offset];
}

Node_SOA* Node_SOA::GetChildNode(ui offset) const {
  assert(offset < branch_count);
  return (Node_SOA*)((char*)this+child_offset[offset]);
}

Point Node_SOA::GetPoint(ui offset) const {
  assert(offset < GetNumberOfDims()*2*GetNumberOfLeafNodeDegrees());
  return points[offset];
}

Point Node_SOA::GetBranchPoint(ui branch_offset, ui dim) const {
  auto offset = dim*GetNumberOfLeafNodeDegrees()+branch_offset;
  assert(offset < GetNumberOfDims()*2*GetNumberOfLeafNodeDegrees());
  return points[offset];
}

void Node_SOA::SetPoint(ui offset, Point point) {
  assert(offset < GetNumberOfDims()*2*GetNumberOfLeafNodeDegrees());
  points[offset] = point;
}

void Node_SOA::SetBranchPoint(ui branch_offset, Point point, ui dim) {
  auto offset = dim*GetNumberOfLeafNodeDegrees() + branch_offset;
  assert(offset < GetNumberOfDims()*2*GetNumberOfLeafNodeDegrees());
  points[offset] = point;
}

void Node_SOA::SetIndex(ui offset, ll _index) {
  assert(offset < GetNumberOfLeafNodeDegrees());
  index[offset] = _index;
}

void Node_SOA::SetChildOffset(ui offset, ll _child_offset) {
  assert(offset < GetNumberOfLeafNodeDegrees());
  child_offset[offset] = _child_offset;
}

void Node_SOA::SetNodeType(NodeType type) {
  assert(type);
  node_type = type;
}

void Node_SOA::SetLevel(int _level) {
  level = _level;
}

void Node_SOA::SetBranchCount(ui _branch_count) {
  assert(_branch_count <= GetNumberOfLeafNodeDegrees());
  branch_count = _branch_count;
  assert(branch_count);
}

__both__  
bool Node_SOA::IsOverlap(Point* query, ui child_offset) {

  for(ui range(lower_boundary, 0, GetNumberOfDims())) {
    ui upper_boundary = lower_boundary+GetNumberOfDims();

    ui node_soa_lower_boundary = lower_boundary*GetNumberOfLeafNodeDegrees()+child_offset;
    ui node_soa_upper_boundary = upper_boundary*GetNumberOfLeafNodeDegrees()+child_offset;

    // Either the query's lower boundary is greather than node's upper boundary
    // or query's upper boundary is less than node's lower boundary, returns
    // false
    if(query[lower_boundary] > points[node_soa_upper_boundary] ||
        query[upper_boundary] < points[node_soa_lower_boundary]) { 
      return false; 
    }
  } 

  return true; 
}

// Get a string representation
std::ostream &operator<<(std::ostream &os, const Node_SOA &node_soa) {
  os << std::fixed << std::setprecision(6);

  os << " Node : " << std::endl;
  os << " NodeType = " << NodeTypeToString(node_soa.GetNodeType()) << std::endl;
  os << " NodeLevel = " << node_soa.GetLevel() << std::endl;
  os << " Branch Count = " << node_soa.branch_count << std::endl;

  for( ui range(i, 0, node_soa.branch_count)) {
    os << " Branch["<< i << "] : " << std::endl;

    for( ui range(d, 0, 2*GetNumberOfDims())) {
      os << " Point[" << d << "] : " << node_soa.points[(d*GetNumberOfLeafNodeDegrees())+i] << std::endl;
    }

    os << " index : " << node_soa.index[i] << std::endl;
    os << " child offset: " << node_soa.child_offset[i] << std::endl;
  }

  return os;
}


} // End of node namespace
} // End of ursus namespace
