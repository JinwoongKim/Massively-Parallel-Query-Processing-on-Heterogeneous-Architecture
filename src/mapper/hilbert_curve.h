#pragma once

#include "mapper.h"

namespace ursus {
namespace mapper {

class Hilbert_Curve : Mapper {
  public:
    unsigned long long 
    MappingIntoSingle(Point points);

    Point
    MappingIntoMulti(unsigned number_of_dimensions,
                     unsigned number_of_bits,
                     unsigned long long index);
  private:
    static bitmask_t 
    bitTranspose(unsigned number_of_dimensions, 
                 unsigned number_of_bits, 
                 bitmask_t inCoords);

    static bitmask_t 
    getIntBits(unsigned number_of_dimensions, 
               unsigned number_of_bytes, 
               char const* c, 
               unsigned y);

    static int 
    hilbert_cmp(unsigned number_of_dimensions, 
                unsigned number_of_bytes, 
                unsigned number_of_bits, 
                void const* coord1, 
                void const* coord2);

    static int 
    hilbert_cmp_work(unsigned number_of_dimensions, 
                     unsigned number_of_bytes, 
                     unsigned number_of_bits, 
                     unsigned max, 
                     unsigned y, 
                     char const* c1, 
                     char const* c2, 
                     unsigned rotation, 
                     bitmask_t bits, 
                     bitmask_t index, 
                     BitReader getBits);
};

} // End of mapper namespace
} // End of ursus namespace
