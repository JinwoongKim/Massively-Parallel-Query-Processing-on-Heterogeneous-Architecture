#pragma once

#include "point.h"

namespace ursus {

class Rect {
 public:
  unsigned int GetDims(void) const;
  unsigned int GetBits(void) const;
  __host__ __device__ bool Overlap(struct Rect *r);

 private:
  Point boundary[2];
  // TODO :: Point MinBoundary, MaxBoundary;
};

} // End of ursus namespace


