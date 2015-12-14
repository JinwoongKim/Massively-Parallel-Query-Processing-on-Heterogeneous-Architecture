#include "common/rect.h"
#include <cassert>

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

//float IntersectedRectArea(struct Rect *r1, struct Rect *r2)
//{
//  int i,j;
//
//  float area = 1.0f;
//
//  for( i=0; i<NUMDIMS; i++)
//  {
//    j=i+NUMDIMS;
//    area *= min( r1->boundary[j], r2->boundary[j])-max(r1->boundary[i], r2->boundary[i]);
//  }
//  return area;
//}

} // End of ursus namespace
