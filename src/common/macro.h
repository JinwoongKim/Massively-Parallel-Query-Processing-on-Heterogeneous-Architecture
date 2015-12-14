// For convenience
#define range(i,a,b) i = (a); i < (b); ++i

/**
 * Marcos for hilbert curve
 */
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


