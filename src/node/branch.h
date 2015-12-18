#pragma once

#include "common/global.h"
#include "common/types.h"

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
  void SetMBB(Point* points);

  __host__ __device__ void SetIndex(const unsigned long long index);

  void SetChild(Node* child);


  Point GetPoint(const unsigned int position) const;

  std::vector<Point> GetPoints(void) const;

  __host__ __device__ unsigned long long GetIndex(void) const;

  Node* GetChild(void) const;


  // Get a string representation for debugging
  friend std::ostream &operator<<(std::ostream &os, const Branch &branch);

  friend __host__ __device__ bool operator<(const Branch &lhs, const Branch &rhs);

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  // Minimum Bounding Box
  Point point[GetNumberOfDims()*2];

  //Index to avoid re-visiting 
  unsigned long long index;

  // child pointers 
  Node* child = nullptr;
};

} // End of node namespace
} // End of ursus namespace
