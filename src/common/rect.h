#pragma once

namespace ursus {

//FIXME load below variables from somewhere ...
extern number_of_dimensions;

class Rect {
  private:
    float boundary[2*number_of_dimensions];
};

} // End of ursus namespace
