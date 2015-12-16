#pragma once

#include "common/types.h"

#include <iostream>
#include <vector>
#include <utility>

namespace ursus {
namespace node {

class Rect {
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  Rect(){};

  Rect(std::vector<Point> _points) {
    SetPoints(_points);
  }

  // Copy constructor
  Rect(const Rect& other) { 
    points = other.points;
  }

 //===--------------------------------------------------------------------===//
 // Accessor
 //===--------------------------------------------------------------------===//
  void SetPoints(std::vector<Point> _points);

  Point GetPoint(const unsigned int position) const;

  std::vector<Point> GetPoints(void) const;

 //===--------------------------------------------------------------------===//
 // Function
 //===--------------------------------------------------------------------===//
  bool Overlap(Rect& rect);

  // Get a string representation for debugging
  friend std::ostream &operator<<(std::ostream &os, const Rect &rect);

 private:

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//

  // a harf of them shows lower boundary,
  // the rest part represents upper boundary
  std::vector<Point> points;

};

} // End of node namespace
} // End of ursus namespace
