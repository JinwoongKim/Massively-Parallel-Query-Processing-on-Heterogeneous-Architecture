#pragma once

namespace ursus {

//===--------------------------------------------------------------------===//
// Hilbert Curve
//===--------------------------------------------------------------------===//

/* define the bitmask_t type as an integer of sufficient size */
  //TODO :: Rename bitmask_t to another one
typedef unsigned long long bitmask_t;
/* define the halfmask_t type as an integer of 1/2 the size of bitmask_t */
typedef unsigned long halfmask_t;

#define adjust_rotation(rotation,nDims,bits)                \
  do {                                                      \
    /* rotation = (rotation + 1 + ffs(bits)) % nDims; */    \
    bits &= -bits & nd1Ones;                                \
    while (bits)                                            \
    bits >>= 1, ++rotation;                                 \
    if ( ++rotation >= nDims )                              \
    rotation -= nDims;                                      \
  } while (0)

#define ones(T,k) ((((T)2) << (k-1)) - 1)

#define rdbit(w,k) (((w) >> (k)) & 1)

#define rotateRight(arg, nRots, nDims)                                  \
	((((arg) >> (nRots)) | ((arg) << ((nDims)-(nRots)))) & ones(bitmask_t,nDims))

#define rotateLeft(arg, nRots, nDims)                                   \
	((((arg) << (nRots)) | ((arg) >> ((nDims)-(nRots)))) & ones(bitmask_t,nDims))

#if defined(sparc)
#define __BIG_ENDIAN__
#endif

#if defined(__BIG_ENDIAN__)
#define whichByte(nBytes,y) (nBytes-1-y/8)
#define setBytes(dst,pos,nBytes,val) \
	memset(&dst[pos+1],val,nBytes-pos-1)
#else
#define whichByte(nBytes,y) (y/8)
#define setBytes(dst,pos,nBytes,val) \
	memset(&dst[0],val,pos)
#endif

/*
 * Readers and writers of bits
 */

typedef bitmask_t (*BitReader) (unsigned nDims, unsigned nBytes, char const* c, unsigned y);
typedef void (*BitWriter) (unsigned d, unsigned nBytes, char* c, unsigned y, int fold);






} // End of ursus namespace
