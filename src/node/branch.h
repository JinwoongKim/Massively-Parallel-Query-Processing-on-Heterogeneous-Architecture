#pragma once

#include "common/config.h"
#include "common/types.h"

#include <iostream>
#include <vector>

namespace ursus {
namespace node {

class Branch {
 public:
 //===--------------------------------------------------------------------===//
 // Constructor
 //===--------------------------------------------------------------------===//
 __both__ Branch(){};

 __both__ Branch(const Branch& branch);

 //===--------------------------------------------------------------------===//
 // Accessors
 //===--------------------------------------------------------------------===//
  std::vector<Point> GetPoints(void) const;
  __both__ Point GetPoint(const ui position) const;
  __both__ ll GetIndex(void) const;
  __both__ ll GetChildOffset(void) const;

  void SetRect(Point* point);
  __both__ void SetPoint(Point point, const ui offset);
  __both__ void SetIndex(const ll index);
  __both__ void SetChildOffset(const ll child_offset);

  // Get a string representation for debugging
  friend std::ostream &operator<<(std::ostream &os, const Branch &branch);

  friend __both__ bool operator<(const Branch &lhs, const Branch &rhs);

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  // Minimum Bounding Box
  Point points[GetNumberOfDims()*2];

  // Index to avoid re-visiting 
  ll index;

  // Child offset from current node
  ll child_offset;
};

} // End of node namespace
} // End of ursus namespace
