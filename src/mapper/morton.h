#pragma once

#include "common/types.h"

namespace ursus {

class Mapper{
};

//Morton Code
unsigned int expandBits(unsigned int v);
unsigned int morton3D(float x, float y, float z);
inline unsigned long long splitBy3(unsigned long long a);
inline unsigned long long splitByDim(unsigned long long a, const unsigned dim);
inline unsigned long long mortonEncode_magicbits(unsigned long long x, unsigned long long y, unsigned long long z);
inline unsigned long long mortonEncode_magicbits2(unsigned long long *input, const unsigned int dim);

}  // End ursus namespace
