#pragma once

#include "common/types.h"
#include "common/point.h"

namespace ursus {
namespace mapper {

class Mapper{
  public:
    virtual unsigned long long 
    MappingIntoSingle(Point points);

    virtual Point
    MappingIntoMulti(unsigned number_of_dimensions,
                     unsigned number_of_bits,
                     unsigned long long index);
};

} // End of mapper namespace
} // End of ursus namespace
