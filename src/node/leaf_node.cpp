#include "node/leaf_node.h"

#include "common/macro.h"
#include "node/branch.h"

#include <cassert>

namespace ursus {
namespace node {

//===--------------------------------------------------------------------===//
// Accessor
//===--------------------------------------------------------------------===//

/**
 *@brief : return current node's minimum bounding rectangle
 */
__both__  
std::vector<Point> LeafNode::GetMBB() const {
  std::vector<Point> mbb(GetNumberOfDims()*2);

  for(ui range(dim, 0, GetNumberOfDims())) {
    ui high_dim = dim+GetNumberOfDims();

    float lower_boundary[GetNumberOfLeafNodeDegrees()];
    float upper_boundary[GetNumberOfLeafNodeDegrees()];

    for( ui range(thread, 0, GetNumberOfLeafNodeDegrees())) {
      if( thread < branch_count){
        lower_boundary[ thread ] = branches[thread].GetPoint(dim);
        upper_boundary[ thread ] = branches[thread].GetPoint(high_dim);
      } else {
        lower_boundary[ thread ] = 1.0f;
        upper_boundary[ thread ] = 0.0f;
      }
    }

    //threads in half get lower boundary

    int N = GetNumberOfLeafNodeDegrees()/2 + GetNumberOfLeafNodeDegrees()%2;
    while(N > 1){
      for( ui range(thread, 0, N)) {
        if(lower_boundary[thread] > lower_boundary[thread+N])
          lower_boundary[thread] = lower_boundary[thread+N];
      }
      N = N/2 + N%2;
    }
    if(N==1) {
      if( lower_boundary[0] > lower_boundary[1])
        lower_boundary[0] = lower_boundary[1];
    }
    //other half threads get upper boundary
    N = GetNumberOfLeafNodeDegrees()/2 + GetNumberOfLeafNodeDegrees()%2;
    while(N > 1){
      for( ui range(thread, 0, N )) {
        if(upper_boundary[thread] < upper_boundary[thread+N])
          upper_boundary[thread] = upper_boundary[thread+N];
      }
      N = N/2 + N%2;
    }
    if(N==1) {
      if ( upper_boundary[0] < upper_boundary[1] )
          upper_boundary[0] = upper_boundary[1];
    }

    mbb[dim] = lower_boundary[0];
    mbb[high_dim] = upper_boundary[0];
  }
  return mbb;
}

__both__
Branch LeafNode::GetBranch(ui offset) const {
  assert(offset < branch_count);
  return branches[offset];
}

__both__
ui LeafNode::GetBranchCount(void) const {
  return branch_count;
}

__both__
Point LeafNode::GetBranchPoint(ui branch_offset, ui point_offset) const{
  return branches[branch_offset].GetPoint(point_offset);
}

__both__
ll LeafNode::GetBranchIndex(ui branch_offset) const{
  return branches[branch_offset].GetIndex();
}

__both__
ll LeafNode::GetLastBranchIndex(void) const{
  return branches[branch_count-1].GetIndex();
}

__both__
ll LeafNode::GetBranchChildOffset(ui branch_offset) const{
  return branches[branch_offset].GetChildOffset();
}

__both__
LeafNode* LeafNode::GetBranchChildLeafNode(ui branch_offset) const{
  return (LeafNode*)((char*)this+branches[branch_offset].GetChildOffset());
}

__both__
NodeType LeafNode::GetNodeType(void) const {
  return node_type;
}

__both__
int LeafNode::GetLevel(void) const {
  return level;
}

__both__
void LeafNode::SetBranch(Branch _branch, ui offset) {
  branches[offset] = _branch;
}

__both__
void LeafNode::SetBranchCount(ui _branch_count) {
  branch_count = _branch_count;
  assert(branch_count);
}

__both__
void LeafNode::SetBranchPoint(ui branch_offset, Point point, 
                          ui point_offset) {
  branches[branch_offset].SetPoint(point, point_offset);
}

__both__
void LeafNode::SetBranchIndex(ui branch_offset, ll index) {
  branches[branch_offset].SetIndex(index);
}

__both__
void LeafNode::SetBranchChildOffset(ui branch_offset, ll child_offset) {
  branches[branch_offset].SetChildOffset(child_offset);
}

__both__
void LeafNode::SetNodeType(NodeType type) {
  assert(type);
  node_type = type;
}

__both__
void LeafNode::SetLevel(int _level) {
  level = _level;
}

bool LeafNode::IsOverlap(Point* query, ui branch_offset) {

  for(ui range(lower_boundary, 0, GetNumberOfDims())) {
    int upper_boundary = lower_boundary+GetNumberOfDims();  

    if (query[lower_boundary] > branches[branch_offset].GetPoint(upper_boundary) ||
        query[upper_boundary] < branches[branch_offset].GetPoint(lower_boundary)) {
      return false;
    }
  }

  return true;
}

// This is for disjoint BVH
bool LeafNode::IsOverlap(ui branch_offset, ui branch_offset2) {

  for(ui range(lower_boundary, 0, GetNumberOfDims())) {
    int upper_boundary = lower_boundary+GetNumberOfDims();  

    if (branches[branch_offset].GetPoint(lower_boundary) > branches[branch_offset2].GetPoint(upper_boundary) ||
        branches[branch_offset].GetPoint(upper_boundary) < branches[branch_offset2].GetPoint(lower_boundary)) {
      return false;
    }
  }

  return true;
}

// Get a string representation
std::ostream &operator<<(std::ostream &os, const LeafNode &node) {
  os << " LeafNode : " << std::endl;
  os << " NodeType = " << NodeTypeToString(node.GetNodeType()) << std::endl;
  os << " LeafNodeLevel = " << node.GetLevel() << std::endl;
  os << " Branch Count = " << node.GetBranchCount() << std::endl;
  for( ui range(i, 0, node.GetBranchCount())) {
    os << " Branch["<< i << "] : " << node.GetBranch(i) << std::endl;
  }
  return os;
}

} // End of node namespace
} // End of ursus namespace
