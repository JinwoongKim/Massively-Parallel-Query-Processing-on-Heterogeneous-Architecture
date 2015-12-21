#pragma once

#include "node/node.h"

#include "common/global.h"

#include <iostream>

namespace ursus {
namespace node {

class Branch {
 public:
 //===--------------------------------------------------------------------===//
 // Constructor
 //===--------------------------------------------------------------------===//
 __host__ __device__ Branch(){};

 __host__ __device__ Branch(const Branch& branch);

 //===--------------------------------------------------------------------===//
 // Accessors
 //===--------------------------------------------------------------------===//
  void SetMBB(Point_Ptr point);

  __host__ __device__ void SetIndex(const ull index);

  void SetChild(Node_Ptr child);


  __host__ __device__ Point GetPoint(const ui position) const;

  std::vector<Point> GetPoints(void) const;

  __host__ __device__ ull GetIndex(void) const;

  __host__ __device__ Node_Ptr GetChild(void) const;

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
  ull index;

  // child pointers 
  Node_Ptr child = nullptr;
};

} // End of node namespace
} // End of ursus namespace
