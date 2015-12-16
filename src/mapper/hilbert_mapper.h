#pragma once

#include "common/types.h"

#include <vector>

namespace ursus {
namespace mapper {

class Hilbert_Mapper {
 public:
 static bitmask_t MappingIntoSingle(unsigned int number_of_dimensions,
                                    unsigned int number_of_bits,
                                    std::vector<Point> points);

 static std::vector<Point> MappingIntoMulti(unsigned int number_of_dimensions,
                                            unsigned int number_of_bits,
                                            bitmask_t index);
 private:
  static bitmask_t bitTranspose(unsigned int number_of_dimensions, 
                                unsigned int number_of_bits, 
                                bitmask_t inCoords);

  static bitmask_t getIntBits(unsigned int number_of_dimensions, 
                              unsigned int number_of_bytes, 
                              char const* c, 
                              unsigned int y);

  static int hilbert_cmp(unsigned int number_of_dimensions, 
                         unsigned int number_of_bytes, 
                         unsigned int number_of_bits, 
                         void const* coord1, 
                         void const* coord2);

  static int hilbert_cmp_work(unsigned int number_of_dimensions, 
                              unsigned int number_of_bytes, 
                              unsigned int number_of_bits, 
                              unsigned int max, 
                              unsigned int y, 
                              char const* c1, 
                              char const* c2, 
                              unsigned int rotation, 
                              bitmask_t bits, 
                              bitmask_t index, 
                              BitReader getBits);
};


} // End of mapper namespace
} // End of ursus namespace
