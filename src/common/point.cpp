#include "common/point.h"
#include <cassert>

// TODO :: Move out to common.h
#define range(i,a,b) i = (a); i < (b); ++i

namespace ursus {

unsigned int Point::GetDims(void) const{ 
  return number_of_dimensions; 
}

unsigned int Point::GetBits(void) const{ 
  return number_of_bits; 
}

unsigned long long* Point::GetPoints(void) const{
  return points; 
}

unsigned long long Point::GetPoint(unsigned int position) const{
  assert(position >= 0  || position < GetDims());
  return points[position]; 
}

void Point::SetDims(unsigned int _number_of_dimensions) { 
  assert( _number_of_dimensions > 0 );
  number_of_dimensions = _number_of_dimensions; 
}

void Point::SetBits(unsigned int _number_of_bits) { 
  assert( _number_of_bits > 0 );
  number_of_bits = _number_of_bits; 
}

void Point::SetPoints(unsigned long long* _points) { 
  assert( _points != nullptr );
  points = _points; 
}

bool operator> (Point &p1, Point &p2) {
  assert(p1.GetDims() == p2.GetDims());
  unsigned int number_of_dimensions = p1.GetDims();

 for(unsigned int range ( dim, 0, number_of_dimensions )) {
   if( p1.GetPoint(dim) <= p2.GetPoint(dim))  {
     return false;
   }
 }
 return true;
}
     
bool operator< (Point &p1, Point &p2) {
  assert(p1.GetDims() == p2.GetDims());
  unsigned int number_of_dimensions = p1.GetDims();

 for(unsigned int range ( dim, 0, number_of_dimensions )) {
   if( p1.GetPoint(dim) >= p2.GetPoint(dim))  {
     return false;
   }
 }
 return true;
}

} // End of ursus namespace
