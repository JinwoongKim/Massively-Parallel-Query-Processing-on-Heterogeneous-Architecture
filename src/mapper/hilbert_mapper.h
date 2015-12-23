#pragma once

#include "common/types.h"

#include <vector>

namespace ursus {
namespace mapper {

class Hilbert_Mapper {
 public:
 static ull MappingIntoSingle(ui number_of_dimensions,
                              ui number_of_bits,
                              std::vector<Point> points);

 static std::vector<Point> MappingIntoMulti(ui number_of_dimensions,
                                            ui number_of_bits,
                                            ull index);
 private:
  static ull bitTranspose(ui number_of_dimensions, 
                          ui number_of_bits, 
                          ull inCoords);

  static ull getIntBits(ui number_of_dimensions, 
                        ui number_of_bytes, 
                        char const* c, ui y);

  static int hilbert_cmp(ui number_of_dimensions, 
                         ui number_of_bytes, 
                         ui number_of_bits, 
                         void const* coord1, void const* coord2);

  static int hilbert_cmp_work(ui number_of_dimensions, 
                              ui number_of_bytes, 
                              ui number_of_bits, 
                              ui max, ui y, 
                              char const* c1, char const* c2, 
                              ui rotation, ull bits, ull index, 
                              BitReader getBits);
};


} // End of mapper namespace
} // End of ursus namespace
