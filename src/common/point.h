#pragma once

namespace ursus {

class Point{
  public:
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
    // ACCESSORS
    //===--------------------------------------------------------------------===//

    unsigned int GetDims(void) const{ return number_of_dimensions; }

    unsigned int GetBits(void) const{ return number_of_bits; }

    unsigned long long* GetPoints(void) const{ return points; }

  private:
    unsigned number_of_dimensions;
    unsigned number_of_bits;

    unsigned long long* points;
};

} // End of ursus namespace
