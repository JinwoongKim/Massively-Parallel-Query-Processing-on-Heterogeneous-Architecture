#include "common/point.h"

#include <cassert>

namespace ursus {

unsigned int 
Point::GetDims(void) const{ 
  return number_of_dimensions; 
}

unsigned int 
Point::GetBits(void) const{ 
  return number_of_bits; 
}

unsigned long long* 
Point::GetPoints(void) const{
  return points; 
}

void 
Point::SetDims(unsigned int _number_of_dimensions) { 
  assert( _number_of_dimensions > 0 );
  number_of_dimensions = _number_of_dimensions; 
}

void 
Point::SetBits(unsigned int _number_of_bits) { 
  assert( _number_of_bits > 0 );
  number_of_bits = _number_of_bits; 
}

void 
Point::SetPoints(unsigned long long* _points) { 
  //FIXME
  //assert( _points != nullptr );
  points = _points; 
}

} // End of ursus namespace
