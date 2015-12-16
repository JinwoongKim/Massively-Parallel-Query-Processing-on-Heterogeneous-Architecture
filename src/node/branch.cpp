#include "node/node.h"

#include <cassert>
#include <utility>

namespace ursus {
namespace node {

void Branch::SetRect(Rect _rect) {
  rect = _rect;
}

void Branch::SetRect(std::vector<Point> rect_points) {
  rect.SetPoints(rect_points);
}

void Branch::SetIndex(const unsigned long long _index) {
  index = _index;
}

void Branch::SetChild(Node* _child) {
  assert(_child);
  child = _child;
}

Rect Branch::GetRect(void) const {
  return rect;
}

unsigned long long Branch::GetIndex(void) const {
  return index;
}

Node* Branch::GetChild(void) const {
  return child;
}

// Get a string representation
std::ostream &operator<<(std::ostream &os, const Branch &branch) {
  os << " Branch : " << std::endl
     << branch.GetRect() << std::endl
     << " Index = " << branch.GetIndex() << std::endl
     << " Child = " << branch.GetChild() << std::endl;

  return os;
}

} // End of node namespace
} // End of ursus namespace
