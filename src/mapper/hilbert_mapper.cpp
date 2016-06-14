#include "hilbert_mapper.h"
#include "mapper/hilbert_macro.h"

#include "common/macro.h"

namespace ursus {
namespace mapper {

/**
* @brief Convert points of a point on a Hilbert curve to its index.
*        Assumptions : number_of_dimensions*number_of_bits <= (sizeof ll) * (bits_per_byte)
* @param points : n number_of_bits-bit coordinates.
* @return index value : number_of_dimensions*number_of_bits bits.
*/
ll 
HilbertMapper::MappingIntoSingle(ui number_of_dimensions,
                                  ui number_of_bits,
                                  std::vector<Point> points) {
  std::vector<ll> coord(number_of_dimensions);

  for(int range(i, 0, number_of_dimensions)) {
    coord[i] = (ll) (1000000*points[i]);
  }

  if (number_of_dimensions > 1) {
    ui const number_of_dimensionsBits = number_of_dimensions*number_of_bits;
    ll index;
    ui d;
    ll coords = 0;

    for (d = number_of_dimensions; d--; ) {
      coords <<= number_of_bits;
      coords |= coord[d];
    }

    if (number_of_bits > 1) {
      ul const ndOnes = ones(ul,number_of_dimensions); 
      ul const nd1Ones= ndOnes >> 1; /* for adjust_rotation */
      ui b = number_of_dimensionsBits;
      ui rotation = 0;
      ul flipBit = 0;
      ll const nthbits = ones(ll,number_of_dimensionsBits) / ndOnes;
      coords = bitTranspose(number_of_dimensions, number_of_bits, coords);
      coords ^= coords >> number_of_dimensions;
      index = 0;

      do {
        ul bits = (coords >> (b-=number_of_dimensions)) & ndOnes;
        bits = rotateRight(flipBit ^ bits, rotation, number_of_dimensions);
        index <<= number_of_dimensions;
        index |= bits;
        flipBit = (ul)1 << rotation;
        adjust_rotation(rotation,number_of_dimensions,bits);
      } while (b);
      index ^= nthbits >> 1;
    } else {
      index = coords;
    }

    for (d = 1; d < number_of_dimensionsBits; d *= 2) {
      index ^= index >> d;
    }

    return index;
  } else {
    return coord[0];
  }
}

/**
 * @brief Convert an index into a Hilbert curve to a set of points.
 * @param number_of_dimensions : number of coordinate axes.
 * @param number_of_bits : number of bits per axis.
 * @param index  : The index, contains number_of_dimensions*number_of_bits bits
 *                 (so number_of_dimensions*number_of_bits must be <= 8*sizeof(ll)).
 * @return coordinate : the list of number of dimensions points,
 *                      each with number of bits 
 */
std::vector<Point>
HilbertMapper::MappingIntoMulti(ui number_of_dimensions,
                                 ui number_of_bits,
                                 ll index) {
  std::vector<ll> coord(number_of_dimensions);

  if (number_of_dimensions > 1){
    ll coords;
    ul const nbOnes = ones(ul,number_of_bits);
    ui d;

    if (number_of_bits > 1){
      ui const number_of_dimensionsBits = number_of_dimensions*number_of_bits;
      ul const ndOnes = ones(ul,number_of_dimensions);
      ul const nd1Ones= ndOnes >> 1; /* for adjust_rotation */
      ui b = number_of_dimensionsBits;
      ui rotation = 0;
      ul flipBit = 0;
      ll const nthbits = ones(ll,number_of_dimensionsBits) / ndOnes;
      index ^= (index ^ nthbits) >> 1;
      coords = 0;

      do {
        ul bits = (index >> (b-=number_of_dimensions)) & ndOnes;
        coords <<= number_of_dimensions;
        coords |= rotateLeft(bits, rotation, number_of_dimensions) ^ flipBit;
        flipBit = (ul)1 << rotation;
        adjust_rotation(rotation,number_of_dimensions,bits);
      } while (b);

      for (b = number_of_dimensions; b < number_of_dimensionsBits; b *= 2) {
        coords ^= coords >> b;
      }
      coords = bitTranspose(number_of_bits, number_of_dimensions, coords);
    } else {
      coords = index ^ (index >> 1);
    }

    for (d = 0; d < number_of_dimensions; ++d) {
      coord[d] = coords & nbOnes;
      coords >>= number_of_bits;
    }
  }
  else{
    coord[0] = index;
  }

  std::vector<Point> points(number_of_dimensions);

  for( int range(i, 0, number_of_dimensions)) {
    points[i] = (Point)(coord[i]/1000000.0);
  }

  return points;
}

ll
HilbertMapper::bitTranspose(ui number_of_dimensions, 
                             ui number_of_bits, 
                             ll inCoords) {

  ui const number_of_dimensions1 = number_of_dimensions-1;
  ui inB = number_of_bits;
  ui utB;
  ll inFieldEnds = 1;
  ll inMask = ones(ll,inB);
  ll coords = 0;

  while ((utB = inB / 2)) {
    ui const shiftAmt = number_of_dimensions1 * utB;
    ll const utFieldEnds =
      inFieldEnds | (inFieldEnds << (shiftAmt+utB));
    ll const utMask =
      (utFieldEnds << utB) - utFieldEnds;
    ll utCoords = 0;
    ui d;

    if (inB & 1) {
      ll const inFieldStarts = inFieldEnds << (inB-1);
      ui oddShift = 2*shiftAmt;

      for (d = 0; d < number_of_dimensions; ++d) {
        ll in = inCoords & inMask;
        inCoords >>= inB;
        coords |= (in & inFieldStarts) <<	oddShift++;
        in &= ~inFieldStarts;
        in = (in | (in << shiftAmt)) & utMask;
        utCoords |= in << (d*utB);
      }
    } else {
      for (d = 0; d < number_of_dimensions; ++d) {
        ll in = inCoords & inMask;
        inCoords >>= inB;
        in = (in | (in << shiftAmt)) & utMask;
        utCoords |= in << (d*utB);
      }
    }
    inCoords = utCoords;
    inB = utB;
    inFieldEnds = utFieldEnds;
    inMask = utMask;
  }
  coords |= inCoords;
  return coords;
}

ll
HilbertMapper::getIntBits(ui number_of_dimensions, ui number_of_bytes, 
                           char const* c, ui y){
  ui const bit = y%8;
  ui const offs = whichByte(number_of_bytes,y);
  ui d;
  ll bits = 0;
  c += offs;

  for (d = 0; d < number_of_dimensions; ++d) {
    bits |= rdbit(*c, bit) << d;
    c += number_of_bytes;
  }
  return bits;
}

/*****************************************************************
 * hilbert_cmp, hilbert_ieee_cmp
 * 
 * Determine which of two points lies further along the Hilbert curve
 * Inputs:
 *  number_of_dimensions:      Number of points.
 *  number_of_bytes:     Number of bytes of storage/coordinate (hilbert_cmp only)
 *  number_of_bits:      Number of bits/coordinate. (hilbert_cmp only)
 *  coord1:     Array of number_of_dimensions number_of_bytes-byte points (or doubles for ieee_cmp).
 *  coord2:     Array of number_of_dimensions number_of_bytes-byte points (or doubles for ieee_cmp).
 * Return value:
 *      -1, 0, or 1 according to whether
 coord1<coord2, coord1==coord2, coord1>coord2
 * Assumptions:
 *      number_of_bits <= (sizeof ll) * (bits_per_byte)
 */

int
HilbertMapper::hilbert_cmp(ui number_of_dimensions, 
                            ui number_of_bytes, 
                            ui number_of_bits,
                            void const* c1, void const* c2){
  ll const one = 1;
  ll bits = one << (number_of_dimensions-1);
  return hilbert_cmp_work(number_of_dimensions, number_of_bytes, 
                          number_of_bits, 0, number_of_bits, 
                          (char const*)c1, (char const*)c2, 0, bits, bits, getIntBits);
}

int
HilbertMapper::hilbert_cmp_work(ui number_of_dimensions, 
                                 ui number_of_bytes, 
                                 ui number_of_bits,
                                 ui max, ui y,
                                 char const* c1, char const* c2,
                                 ui rotation, ll bits, ll index,
                                 BitReader getBits){

  ll const one = 1;
  ll const nd1Ones = ones(ll,number_of_dimensions) >> 1; /* used in adjust_rotation macro */
  while (y-- > max) {
    ll reflection = getBits(number_of_dimensions, number_of_bytes, c1, y);
    ll diff = reflection ^ getBits(number_of_dimensions, number_of_bytes, c2, y);
    bits ^= reflection;
    bits = rotateRight(bits, rotation, number_of_dimensions);
    if (diff) {
      ui d;
      diff = rotateRight(diff, rotation, number_of_dimensions);
      for (d = 1; d < number_of_dimensions; d *= 2) {
        index ^= index >> d;
        bits  ^= bits  >> d;
        diff  ^= diff  >> d;
      }
      return (((index ^ y ^ number_of_bits) & 1) == (bits < (bits^diff)))? -1: 1;
    }
    index ^= bits;
    reflection ^= one << rotation;
    adjust_rotation(rotation,number_of_dimensions,bits);
    bits = reflection;
  }
  return 0;
}

} // End of mapper namespace
} // End of ursus namespace
