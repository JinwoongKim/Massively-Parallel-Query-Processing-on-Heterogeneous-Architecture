#pragma once
#include "common/point.h"

namespace ursus {

class Rect {
 public:
  unsigned int GetDims(void) const;
  unsigned int GetBits(void) const;
  __host__ __device__ bool Overlap(struct Rect *r);

  //TODO static function or ?? ...
  //float IntersectedRectArea(struct Rect *r1, struct Rect *r2);

 private:
  Point boundary[2];
  // TODO :: Point MinBoundary, MaxBoundary;
};

} // End of ursus namespace


