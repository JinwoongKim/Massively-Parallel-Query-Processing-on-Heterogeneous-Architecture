#pragma once

#include <iostream>

namespace ursus {

class Point{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  Point(unsigned int number_of_dimensions,
        unsigned int number_of_bits) 
  : number_of_dimensions(number_of_dimensions),
    number_of_bits(number_of_bits){

    points = new unsigned long long [number_of_dimensions];
  };
  ~Point(){
    delete points;
  }

 //===--------------------------------------------------------------------===//
 // Accessors
 //===--------------------------------------------------------------------===//
  unsigned int GetDims(void) const;
  unsigned int GetBits(void) const;
  unsigned long long* GetPoints(void) const;
  unsigned long long GetPoint(unsigned int position) const;

  void SetDims(unsigned int number_of_dimensions);
  void SetBits(unsigned int number_of_bits);
  void SetPoints(unsigned long long* points);

 //===--------------------------------------------------------------------===//
 // Operators
 //===--------------------------------------------------------------------===//
  friend bool operator> (Point &p1, Point &p2);
  friend bool operator< (Point &p1, Point &p2);

  // Get a string representation for debugging
  friend std::ostream &operator<<(std::ostream &os, const Point &point);

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  // # of dims
  unsigned number_of_dimensions;
  
  // # of bits which represent each dimension 
  unsigned number_of_bits;

  // real point data 
  unsigned long long* points;
};

} // End of ursus namespace
