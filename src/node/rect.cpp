#include "common/macro.h"
#include "node/rect.h"

#include <cassert>
#include <iomanip>

namespace ursus {
namespace node {

void Rect::SetPoints(std::vector<Point> _points) {
  points = _points;
}

Point Rect::GetPoint(const unsigned int position) const{
  assert(position < points.size());
  return points[position];
}

std::vector<Point> Rect::GetPoints(void) const{
  return points;
}

bool Rect::Overlap(Rect& rect){
  int number_of_dimensions = points.size()/2;

  //TODO use for_each?
  for( int range(lower_bound, 0, number_of_dimensions)) {
    int upper_bound = lower_bound+number_of_dimensions;
    if( points[lower_bound] > rect.GetPoint(upper_bound) ||
        points[upper_bound] < rect.GetPoint(lower_bound)) {
      return false;
    }
  }
  return true;
}

// Get a string representation
std::ostream &operator<<(std::ostream &os, const Rect &rect) {
  os << " Rect : " << std::endl;
  auto points = rect.GetPoints();
  os << std::fixed << std::setprecision( 6 );
  for( auto point : points) {
  os << " Point = " << point << std::endl;
  }

  return os;
}

} // End of node namespace
} // End of ursus namespace
