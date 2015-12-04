#include "common/rect.h"

namespace ursus {

unsigned int Rect::GetDims(void) const{
  return boundary[0].GetDims();
}

unsigned int Rect::GetBits(void) const{
  return boundary[0].GetBits();
}

__host__ __device__ 
bool Rect::Overlap(struct Rect *r){
  assert(GetDims()==r->GetDims());
  assert(GetBits()==r->GetBits());

  // minimum boundary > maximum boundary or maximum boundary < minimum boundary
  if( boundary[0] > r->boundary[1] || boundary[1] < r->boundary[0] ){
    return false;
  }

  return true;
}

} // End of ursus namespace
