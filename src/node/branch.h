#pragma once

#include "common/config.h"
#include "common/types.h"

#include <iostream>
#include <vector>

namespace ursus {
namespace node {

class Node;
typedef Node* Node_Ptr;

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
  __both__ ull GetIndex(void) const;
  __both__ Node_Ptr GetChild(void) const;

  void SetRect(Point* point);
  __both__ void SetPoint(Point point, const ui offset);
  __both__ void SetIndex(const ull index);
  __both__ void SetChild(Node_Ptr child);

  // Get a string representation for debugging
  friend std::ostream &operator<<(std::ostream &os, const Branch &branch);

  friend __both__ bool operator<(const Branch &lhs, const Branch &rhs);

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  // Minimum Bounding Box
  Point points[GetNumberOfDims()*2];

  //Index to avoid re-visiting 
  ull index;

  // child pointers 
  Node_Ptr child = nullptr;
};

} // End of node namespace
} // End of ursus namespace
