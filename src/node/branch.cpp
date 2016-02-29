#include "node/branch.h"

#include "common/macro.h"

#include <cassert>
#include <iomanip>

namespace ursus {
namespace node {

//===--------------------------------------------------------------------===//
// Constructor
//===--------------------------------------------------------------------===//
__both__
Branch::Branch(const Branch& branch)
 : index(branch.GetIndex()), child(branch.GetChild()) { 
   for(ui range(i, 0, GetNumberOfDims()*2)) {
     points[i] = branch.GetPoint(i);
   }
}

//===--------------------------------------------------------------------===//
// Accessors
//===--------------------------------------------------------------------===//
std::vector<Point> Branch::GetPoints(void) const{
  std::vector<Point> point_vec(GetNumberOfDims()*2);
  std::copy(points, points+GetNumberOfDims()*2, point_vec.begin());
  return point_vec;
}

__both__ 
Point Branch::GetPoint(const ui position) const{
  return points[position];
}

__both__ 
unsigned long long Branch::GetIndex(void) const {
  return index;
}

__both__
Node* Branch::GetChild(void) const {
  return child;
}
 
void Branch::SetRect(Point* _points) {
  std::copy(_points, _points+GetNumberOfDims(), points);
  std::copy(_points, _points+GetNumberOfDims(), points+GetNumberOfDims());
}

__both__
void Branch::SetPoint(Point point, const ui offset) {
  assert(offset < GetNumberOfDims()*2);
  points[offset] = point;
}

__both__
void Branch::SetIndex(const ull _index) {
  index = _index;
}

__both__
void Branch::SetChild(Node* _child) {
  assert(_child);
  child = _child;
}

// Get a string representation
std::ostream &operator<<(std::ostream &os, const Branch &branch) {
  os << " Branch : " << std::endl;
  os << " MBB : " << std::endl;
  os << std::fixed << std::setprecision(6);
  for( int range(i, 0, GetNumberOfDims()*2)) {
    os << " Point["<< i << "] : " << branch.points[i] << std::endl;
  }
  os << " Index = " << branch.GetIndex() << std::endl;
  os << " Child = " << branch.GetChild() << std::endl;
  return os;
}
__both__ 
bool operator<(const Branch &lhs, const Branch &rhs) {
  return lhs.GetIndex() < rhs.GetIndex();
}

} // End of node namespace
} // End of ursus namespace
