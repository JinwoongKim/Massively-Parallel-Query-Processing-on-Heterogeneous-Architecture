#pragma once

#include "node/rect.h"

#include <iostream>
#include <vector>

namespace ursus {
namespace node {

class Node;

class Branch {
 public:
 //===--------------------------------------------------------------------===//
 // Accessors
 //===--------------------------------------------------------------------===//
  void SetRect(Rect rect);

  void SetRect(std::vector<Point> rect_points);

  void SetIndex(const unsigned long long index);

  void SetChild(Node* child);


  Rect GetRect(void) const;

  unsigned long long GetIndex(void) const;

  Node* GetChild(void) const;

  // Get a string representation for debugging
  friend std::ostream &operator<<(std::ostream &os, const Branch &branch);

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  // Minimum Bounding Box
  Rect rect;

  //Index to avoid re-visiting 
  unsigned long long index;

  // child pointers 
  Node* child; 
};

} // End of node namespace
} // End of ursus namespace
