#include "common/macro.h"
#include "node/node_soa.h"

#include <cassert>

namespace ursus {
namespace node {

NodeType Node_SOA::GetNodeType(void) const {
  return node_type;
}
int Node_SOA::GetLevel(void) const {
  return level;
}

void Node_SOA::SetPoint(ui offset, Point point) {
  assert(offset < GetNumberOfDims()*2*GetNumberOfDegrees());
  points[offset] = point;
}

void Node_SOA::SetIndex(ui offset, ull _index) {
  assert(offset < GetNumberOfDegrees());
  index[offset] = _index;
}

void Node_SOA::SetChild(ui offset, Node_SOA* _child) {
  assert(offset < GetNumberOfDegrees());
  child[offset] = _child;
}

void Node_SOA::SetNodeType(NodeType type) {
  assert(type);
  node_type = type;
}

void Node_SOA::SetLevel(int _level) {
  level = _level;
}

void Node_SOA::SetBranchCount(ui _branch_count) {
  assert(_branch_count <= GetNumberOfDegrees());
  branch_count = _branch_count;
}

// Get a string representation
std::ostream &operator<<(std::ostream &os, const Node_SOA &node_soa) {
  os << " Node : " << std::endl;
  os << " NodeType = " << NodeTypeToString(node_soa.GetNodeType()) << std::endl;
  os << " NodeLevel = " << node_soa.GetLevel() << std::endl;
  os << " Branch Count = " << node_soa.branch_count << std::endl;

  for( ui range(i, 0, node_soa.branch_count)) {
    os << " Branch["<< i << "] : " << std::endl;

    for( ui range(d, 0, 2*GetNumberOfDims())) {
      os << " Point[" << d << "] : " << node_soa.points[(d*GetNumberOfDegrees())+i] << std::endl;
    }

    os << " index : " << node_soa.index[i] << std::endl;
    os << " child : " << node_soa.child[i] << std::endl;
  }

  return os;
}


} // End of node namespace
} // End of ursus namespace
