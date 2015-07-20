#include <stdio.h>
#include <limits.h>



/* C header file for Hilbert curve functions */
#if !defined(_hilbert_h_)
#define _hilbert_h_

#ifdef __cplusplus
extern "C" {
#endif

/* define the bitmask_t type as an integer of sufficient size */
typedef unsigned long long bitmask_t;
/* define the halfmask_t type as an integer of 1/2 the size of bitmask_t */
typedef unsigned long halfmask_t;

/*****************************************************************
 * hilbert_i2c
 * 
 * Convert an index into a Hilbert curve to a set of coordinates.
 * Inputs:
 *  nDims:      Number of coordinate axes.
 *  nBits:      Number of bits per axis.
 *  index:      The index, contains nDims*nBits bits (so nDims*nBits must be <= 8*sizeof(bitmask_t)).
 * Outputs:
 *  coord:      The list of nDims coordinates, each with nBits bits.
 * Assumptions:
 *      nDims*nBits <= (sizeof index) * (bits_per_byte)
 */

extern void hilbert_i2c(unsigned nDims, unsigned nBits, bitmask_t index, bitmask_t coord[]);

/*****************************************************************
 * hilbert_c2i
 * 
 * Convert coordinates of a point on a Hilbert curve to its index.
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBits:      Number of bits/coordinate.
 *  coord:      Array of n nBits-bit coordinates.
 * Outputs:
 *  index:      Output index value.  nDims*nBits bits.
 * Assumptions:
 *      nDims*nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */

bitmask_t hilbert_c2i(unsigned nDims, unsigned nBits, bitmask_t const coord[]);

/*****************************************************************
 * hilbert_cmp, hilbert_ieee_cmp
 * 
 * Determine which of two points lies further along the Hilbert curve
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBytes:     Number of bytes of storage/coordinate (hilbert_cmp only)
 *  nBits:      Number of bits/coordinate. (hilbert_cmp only)
 *  coord1:     Array of nDims nBytes-byte coordinates (or doubles for ieee_cmp).
 *  coord2:     Array of nDims nBytes-byte coordinates (or doubles for ieee_cmp).
 * Return value:
 *      -1, 0, or 1 according to whether
           coord1<coord2, coord1==coord2, coord1>coord2
 * Assumptions:
 *      nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */

int hilbert_cmp(unsigned nDims, unsigned nBytes, unsigned nBits, void const* coord1, void const* coord2);
int hilbert_ieee_cmp(unsigned nDims, double const* coord1, double const* coord2);

/*****************************************************************
 * hilbert_box_vtx
 * 
 * Determine the first or last vertex of a box to lie on a Hilbert curve
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBytes:     Number of bytes/coordinate.
 *  nBits:      Number of bits/coordinate. (hilbert_cmp only)
 *  findMin:    Is it the least vertex sought?
 *  coord1:     Array of nDims nBytes-byte coordinates - one corner of box
 *  coord2:     Array of nDims nBytes-byte coordinates - opposite corner
 * Output:
 *      c1 and c2 modified to refer to selected corner
 *      value returned is log2 of size of largest power-of-two-aligned box that
 *      contains the selected corner and no other corners
 * Assumptions:
 *      nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */
unsigned
hilbert_box_vtx(unsigned nDims, unsigned nBytes, unsigned nBits,
		int findMin, void* c1, void* c2);
unsigned
hilbert_ieee_box_vtx(unsigned nDims,
		     int findMin, double* c1, double* c2);

/*****************************************************************
 * hilbert_box_pt
 * 
 * Determine the first or last point of a box to lie on a Hilbert curve
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBytes:     Number of bytes/coordinate.
 *  nBits:      Number of bits/coordinate.
 *  findMin:    Is it the least vertex sought?
 *  coord1:     Array of nDims nBytes-byte coordinates - one corner of box
 *  coord2:     Array of nDims nBytes-byte coordinates - opposite corner
 * Output:
 *      c1 and c2 modified to refer to least point
 * Assumptions:
 *      nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */
unsigned
hilbert_box_pt(unsigned nDims, unsigned nBytes, unsigned nBits,
	       int findMin, void* coord1, void* coord2);
unsigned
hilbert_ieee_box_pt(unsigned nDims,
		    int findMin, double* c1, double* c2);

/*****************************************************************
 * hilbert_nextinbox
 * 
 * Determine the first point of a box after a given point to lie on a Hilbert curve
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBytes:     Number of bytes/coordinate.
 *  nBits:      Number of bits/coordinate.
 *  findPrev:   Is the previous point sought?
 *  coord1:     Array of nDims nBytes-byte coordinates - one corner of box
 *  coord2:     Array of nDims nBytes-byte coordinates - opposite corner
 *  point:      Array of nDims nBytes-byte coordinates - lower bound on point returned
 *  
 * Output:
      if returns 1:
 *      c1 and c2 modified to refer to least point after "point" in box
      else returns 0:
        arguments unchanged; "point" is beyond the last point of the box
 * Assumptions:
 *      nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */
int
hilbert_nextinbox(unsigned nDims, unsigned nBytes, unsigned nBits,
		  int findPrev, void* coord1, void* coord2,
		  void const* point);

/*****************************************************************
 * hilbert_incr
 * 
 * Advance from one point to its successor on a Hilbert curve
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBits:      Number of bits/coordinate.
 *  coord:      Array of nDims nBits-bit coordinates.
 * Output:
 *  coord:      Next point on Hilbert curve
 * Assumptions:
 *      nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */

void
hilbert_incr(unsigned nDims, unsigned nBits, bitmask_t coord[]);

#ifdef __cplusplus
}
#endif

//
// Modified R-Tree Structure using CUDA
// UNIST, Ulsan, Rep. of Korea.
//
// 
// Last Modified Date : 2012. 5. 9.


//**********************************************************************
//***************** HIGHT DIMENSIONAL HILBERT CURVE ********************
//**********************************************************************

#define adjust_rotation(rotation,nDims,bits)                            \
	do {                                                                    \
		/* rotation = (rotation + 1 + ffs(bits)) % nDims; */              \
		bits &= -bits & nd1Ones;                                          \
		while (bits)                                                      \
		bits >>= 1, ++rotation;                                         \
		if ( ++rotation >= nDims )                                        \
		rotation -= nDims;                                              \
	} while (0)

#define ones(T,k) ((((T)2) << (k-1)) - 1)

#define rdbit(w,k) (((w) >> (k)) & 1)

#define rotateRight(arg, nRots, nDims)                                  \
	((((arg) >> (nRots)) | ((arg) << ((nDims)-(nRots)))) & ones(bitmask_t,nDims))

#define rotateLeft(arg, nRots, nDims)                                   \
	((((arg) << (nRots)) | ((arg) >> ((nDims)-(nRots)))) & ones(bitmask_t,nDims))


#define DLOGB_BIT_TRANSPOSE
	static bitmask_t
	bitTranspose(unsigned nDims, unsigned nBits, bitmask_t inCoords)
#if defined(DLOGB_BIT_TRANSPOSE)
{
	unsigned const nDims1 = nDims-1;
	unsigned inB = nBits;
	unsigned utB;
	bitmask_t inFieldEnds = 1;
	bitmask_t inMask = ones(bitmask_t,inB);
	bitmask_t coords = 0;

	while ((utB = inB / 2))
	{
		unsigned const shiftAmt = nDims1 * utB;
		bitmask_t const utFieldEnds =
			inFieldEnds | (inFieldEnds << (shiftAmt+utB));
		bitmask_t const utMask =
			(utFieldEnds << utB) - utFieldEnds;
		bitmask_t utCoords = 0;
		unsigned d;
		if (inB & 1)
		{
			bitmask_t const inFieldStarts = inFieldEnds << (inB-1);
			unsigned oddShift = 2*shiftAmt;
			for (d = 0; d < nDims; ++d)
			{
				bitmask_t in = inCoords & inMask;
				inCoords >>= inB;
				coords |= (in & inFieldStarts) <<	oddShift++;
				in &= ~inFieldStarts;
				in = (in | (in << shiftAmt)) & utMask;
				utCoords |= in << (d*utB);
			}
		}
		else
		{
			for (d = 0; d < nDims; ++d)
			{
				bitmask_t in = inCoords & inMask;
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
#else
{
	bitmask_t coords = 0;
	unsigned d;
	for (d = 0; d < nDims; ++d)
	{
		unsigned b;
		bitmask_t in = inCoords & ones(bitmask_t,nBits);
		bitmask_t out = 0;
		inCoords >>= nBits;
		for (b = nBits; b--;)
		{
			out <<= nDims;
			out |= rdbit(in, b);
		}
		coords |= out << d;
	}
	return coords;
}
#endif

/*****************************************************************
 * hilbert_i2c
 * 
 * Convert an index into a Hilbert curve to a set of coordinates.
 * Inputs:
 *  nDims:      Number of coordinate axes.
 *  nBits:      Number of bits per axis.
 *  index:      The index, contains nDims*nBits bits
 *              (so nDims*nBits must be <= 8*sizeof(bitmask_t)).
 * Outputs:
 *  coord:      The list of nDims coordinates, each with nBits bits.
 * Assumptions:
 *      nDims*nBits <= (sizeof index) * (bits_per_byte)
 */
void hilbert_i2c(unsigned nDims, unsigned nBits, bitmask_t index, bitmask_t coord[])
{
	if (nDims > 1)
	{
		bitmask_t coords;
		halfmask_t const nbOnes = ones(halfmask_t,nBits);
		unsigned d;

		if (nBits > 1)
		{
			unsigned const nDimsBits = nDims*nBits;
			halfmask_t const ndOnes = ones(halfmask_t,nDims);
			halfmask_t const nd1Ones= ndOnes >> 1; /* for adjust_rotation */
			unsigned b = nDimsBits;
			unsigned rotation = 0;
			halfmask_t flipBit = 0;
			bitmask_t const nthbits = ones(bitmask_t,nDimsBits) / ndOnes;

			index ^= (index ^ nthbits) >> 1;

			coords = 0;

			do
			{
				halfmask_t bits = (index >> (b-=nDims)) & ndOnes;
				coords <<= nDims;
				coords |= rotateLeft(bits, rotation, nDims) ^ flipBit;

				flipBit = (halfmask_t)1 << rotation;
				adjust_rotation(rotation,nDims,bits);
			} while (b);

			for (b = nDims; b < nDimsBits; b *= 2)
			{
				coords ^= coords >> b;
			}
			coords = bitTranspose(nBits, nDims, coords);
		}
		else
			coords = index ^ (index >> 1);

		for (d = 0; d < nDims; ++d)
		{
			coord[d] = coords & nbOnes;
			coords >>= nBits;
		}
	}
	else
		coord[0] = index;
}

//modified version of hilbert_i2c by jwkim
void hilbert_i2c2
(unsigned nDims, unsigned nBits, bitmask_t index, bitmask_t coord[], int X )
{
	if (nDims > 1)
	{
		bitmask_t coords;
		halfmask_t const nbOnes = ones(halfmask_t,nBits);
		unsigned d;

		if (nBits > 1)
		{
			unsigned const nDimsBits = nDims*nBits;
			halfmask_t const ndOnes = ones(halfmask_t,nDims);
			halfmask_t const nd1Ones= ndOnes >> 1; /* for adjust_rotation */
			unsigned b = nDimsBits;
			unsigned rotation = 0;
			halfmask_t flipBit = 0;
			bitmask_t const nthbits = ones(bitmask_t,nDimsBits) / ndOnes;

			index ^= (index ^ nthbits) >> 1;

			coords = 0;

			do
			{
					halfmask_t bits = (index >> (b-=nDims)) & ndOnes;
					coords <<= nDims;

					coords |= rotateLeft(bits, rotation, nDims) ^ flipBit;

					flipBit = (halfmask_t)1 << rotation;
					adjust_rotation(rotation,nDims,bits);
			}while(b);

			for (b = nDims; b < nDimsBits; b *= 2)
			{
				coords ^= coords >> b;
			}

			coords = bitTranspose(nBits, nDims, coords);

		}
		else
			coords = index ^ (index >> 1);

		for (d = 0; d < nDims; ++d)
		{
			coord[d] = coords & nbOnes;
			coords >>= nBits;
		}

		/*
		int num_shift_d0 = 0;
		if( X >= 9 )
			num_shift_d0 = (X/9)*3;
		if( X%9 > 2 ) 
			num_shift_d0++;
		if( X%9 > 4 ) 
			num_shift_d0++;
		if( X%9 > 6 ) 
			num_shift_d0++;

		int num_shift_d1 = 0;
		if( X >= 9 )
			num_shift_d1 = (X/9)*3;
		if( X%9 > 0 ) 
			num_shift_d1++;
		if( X%9 > 5 ) 
			num_shift_d1++;
		if( X%9 > 7 ) 
			num_shift_d1++;

		int num_shift_d2 = 0;
		if( X >= 9 )
			num_shift_d2 = (X/9)*3;
		if( X%9 > 1 ) 
			num_shift_d2++;
		if( X%9 > 3 ) 
			num_shift_d2++;
		if( X%9 > 8 ) 
			num_shift_d2++;

	  unsigned long long max;

		coord[0] = coord[0] >> num_shift_d0;
		coord[0] = coord[0] << num_shift_d0;
		printf("min coord[0] %llu\n", coord[0]);
    printCommonPrefix(coord[0] , 0 , 30, 64);
  	printf("\n\n");
		max = (1 << num_shift_d0)-1;
		coord[0] = max | coord[0];
		printf("max coord[0] %llu\n", coord[0]);
    printCommonPrefix(coord[0] , 0 , 30, 64);
  	printf("\n\n");


		coord[1] = coord[1] >> num_shift_d1;
		coord[1] = coord[1] << num_shift_d1;
		printf("coord[1] %llu\n", coord[1]);
    printCommonPrefix(coord[1] , 0 , 30, 64);
  	printf("\n\n");
		max = (1 << num_shift_d1)-1;
		coord[1] = max | coord[1];
		printf("max coord[1] %llu\n", coord[1]);
    printCommonPrefix(coord[1] , 0 , 30, 64);
  	printf("\n\n");


		coord[2] = coord[2] >> num_shift_d2;
		coord[2] = coord[2] << num_shift_d2;
		printf("coord[2] %llu\n", coord[2]);
    printCommonPrefix(coord[2] , 0 , 30, 64);
  	printf("\n\n");
		max = (1 << num_shift_d2)-1;
		coord[2] = max | coord[2];
		printf("max coord[2] %llu\n", coord[2]);
    printCommonPrefix(coord[2] , 0 , 30, 64);
  	printf("\n\n");

		*/
	}
	else
		coord[0] = index;
}

/*****************************************************************
 * hilbert_c2i
 * 
 * Convert coordinates of a point on a Hilbert curve to its index.
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBits:      Number of bits/coordinate.
 *  coord:      Array of n nBits-bit coordinates.
 * Outputs:
 *  index:      Output index value.  nDims*nBits bits.
 * Assumptions:
 *      nDims*nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */
bitmask_t
hilbert_c2i(unsigned nDims, unsigned nBits, bitmask_t const coord[])
{
	if (nDims > 1)
	{
		unsigned const nDimsBits = nDims*nBits;
		bitmask_t index;
		unsigned d;
		bitmask_t coords = 0;
		for (d = nDims; d--; )
		{
			coords <<= nBits;
			coords |= coord[d];
		}

		if (nBits > 1)
		{
			halfmask_t const ndOnes = ones(halfmask_t,nDims); 
			halfmask_t const nd1Ones= ndOnes >> 1; /* for adjust_rotation */
			unsigned b = nDimsBits;
			unsigned rotation = 0;
			halfmask_t flipBit = 0;
			bitmask_t const nthbits = ones(bitmask_t,nDimsBits) / ndOnes;
			coords = bitTranspose(nDims, nBits, coords);
			coords ^= coords >> nDims;
			index = 0;
			do
			{
				halfmask_t bits = (coords >> (b-=nDims)) & ndOnes;
				bits = rotateRight(flipBit ^ bits, rotation, nDims);
				index <<= nDims;
				index |= bits;
				flipBit = (halfmask_t)1 << rotation;
				adjust_rotation(rotation,nDims,bits);
			} while (b);
			index ^= nthbits >> 1;
		}
		else
			index = coords;

		for (d = 1; d < nDimsBits; d *= 2)
			index ^= index >> d;

		return index;
	}
	else
		return coord[0];
}


// nDims = 2,  nBits = 10?
//modified version of hilbert_c2i by jwkim
bitmask_t
hilbert_c2i2(unsigned nDims, unsigned nBits, bitmask_t const coord[])
{
	if (nDims > 1)
	{
		unsigned const nDimsBits = nDims*nBits; // 20
		bitmask_t index;
		unsigned d;
		bitmask_t coords = 0;
		for (d = nDims; d--; )
		{
			coords <<= nBits;
			coords |= coord[d];
		}
		// what if coord[0] has bits more than nBits??, it may overwrite previous digits
		// coord[d] << 64-nBits and coord[d] >> 64-nBits

		if (nBits > 1)
		{
			halfmask_t const ndOnes = ones(halfmask_t,nDims);  // 3
			halfmask_t const nd1Ones= ndOnes >> 1; /* for adjust_rotation */ // 1
			unsigned b = nDimsBits; //20
			unsigned rotation = 0;
			halfmask_t flipBit = 0;
			bitmask_t const nthbits = ones(bitmask_t,nDimsBits) / ndOnes; // 1048575 / 3 = 349525
			printf("before bitTranspose %lu\n", coords);
			coords = bitTranspose(nDims, nBits, coords);
			printf("after bitTranspose %lu\n", coords);
			coords ^= coords >> nDims;
			index = 0;
			do
			{
				halfmask_t bits = (coords >> (b-=nDims)) & ndOnes;
				bits = rotateRight(flipBit ^ bits, rotation, nDims);
				index <<= nDims;
				index |= bits;
				flipBit = (halfmask_t)1 << rotation;
				adjust_rotation(rotation,nDims,bits);
			} while (b);
			index ^= nthbits >> 1;
		}
		else
			index = coords;

		for (d = 1; d < nDimsBits; d *= 2)
			index ^= index >> d;

		return index;
	}
	else
		return coord[0];
}

/*****************************************************************
 * Readers and writers of bits
 */

typedef bitmask_t (*BitReader) (unsigned nDims, unsigned nBytes,
		char const* c, unsigned y);
typedef void (*BitWriter) (unsigned d, unsigned nBytes,
		char* c, unsigned y, int fold);


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

	static bitmask_t
getIntBits(unsigned nDims, unsigned nBytes, char const* c, unsigned y)
{
	unsigned const bit = y%8;
	unsigned const offs = whichByte(nBytes,y);
	unsigned d;
	bitmask_t bits = 0;
	c += offs;
	for (d = 0; d < nDims; ++d)
	{
		bits |= rdbit(*c, bit) << d;
		c += nBytes;
	}
	return bits;
}

/*****************************************************************
 * hilbert_cmp, hilbert_ieee_cmp
 * 
 * Determine which of two points lies further along the Hilbert curve
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBytes:     Number of bytes of storage/coordinate (hilbert_cmp only)
 *  nBits:      Number of bits/coordinate. (hilbert_cmp only)
 *  coord1:     Array of nDims nBytes-byte coordinates (or doubles for ieee_cmp).
 *  coord2:     Array of nDims nBytes-byte coordinates (or doubles for ieee_cmp).
 * Return value:
 *      -1, 0, or 1 according to whether
 coord1<coord2, coord1==coord2, coord1>coord2
 * Assumptions:
 *      nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */

	static int
hilbert_cmp_work(unsigned nDims, unsigned nBytes, unsigned nBits,
		unsigned max, unsigned y,
		char const* c1, char const* c2,
		unsigned rotation,
		bitmask_t bits,
		bitmask_t index,
		BitReader getBits)
{
	bitmask_t const one = 1;
	bitmask_t const nd1Ones = ones(bitmask_t,nDims) >> 1; /* used in adjust_rotation macro */
	while (y-- > max)
	{
		bitmask_t reflection = getBits(nDims, nBytes, c1, y);
		bitmask_t diff = reflection ^ getBits(nDims, nBytes, c2, y);
		bits ^= reflection;
		bits = rotateRight(bits, rotation, nDims);
		if (diff)
		{
			unsigned d;
			diff = rotateRight(diff, rotation, nDims);
			for (d = 1; d < nDims; d *= 2)
			{
				index ^= index >> d;
				bits  ^= bits  >> d;
				diff  ^= diff  >> d;
			}
			return (((index ^ y ^ nBits) & 1) == (bits < (bits^diff)))? -1: 1;
		}
		index ^= bits;
		reflection ^= one << rotation;
		adjust_rotation(rotation,nDims,bits);
		bits = reflection;
	}
	return 0;
}

	int
hilbert_cmp(unsigned nDims, unsigned nBytes, unsigned nBits,
		void const* c1, void const* c2)
{
	bitmask_t const one = 1;
	bitmask_t bits = one << (nDims-1);
	return hilbert_cmp_work(nDims, nBytes, nBits, 0, nBits,
			(char const*)c1, (char const*)c2,
			0, bits, bits, getIntBits);
}

//end of high dimensional hilbert curve

#endif /* _hilbert_h_ */

void printCommonPrefix(unsigned long long num, int order, int nbits /* the number of bits to print*/ , int valid_bits)
{
	if( order < nbits ) 
	{
		printCommonPrefix(num>>1, ++order, nbits, valid_bits);

		if(order <= (64-valid_bits))
		{
			printf("X");
		}
		else
		{
			if( num % 2 == 0 )
				printf("0");
			else
				printf("1");
		}
	}
}
unsigned long long RightShift(unsigned long long val, int shift)
{
	int loop = shift/30;
	int i;

	for(i=0; i < loop ; i++)
	  val = val >> 30;

	val = val >> shift%30;

	return val;
}
unsigned long long LeftShift(unsigned long long val, int shift)
{
	int loop = shift/30;
	int i;

	for(i=0; i < loop ; i++)
	  val = val << 30;

	val = val << shift%30;

	return val;
}

void ConvertHilbertIndexToBoundingBox(unsigned long long index, int NUMDIMS, int X, float* rect)
{
	int i,j;
	int nbits = X%NUMDIMS;
	unsigned long long coord[NUMDIMS];
	unsigned long long coords[nbits*2][NUMDIMS];

	unsigned long long current_num;
	unsigned long long start_num = LeftShift( RightShift(index ,X), X);
	unsigned long long increments = LeftShift(1 , (X/NUMDIMS)*NUMDIMS) ;
	unsigned long long end_num = start_num | (LeftShift(1 , X) -1);
	for(i=0, current_num = start_num; current_num <= end_num; current_num+=increments, i++)
	{
		hilbert_i2c(NUMDIMS, 20, current_num, coords[i]);
		for(j=0; j<NUMDIMS; j++)
		{
			coords[i][j] = LeftShift(RightShift(coords[i][j] , (X/NUMDIMS)) , (X/NUMDIMS));
		}
	}

  hilbert_i2c(NUMDIMS, 20, index, coord);

	int diff_bits[NUMDIMS];
	for(i=0; i<NUMDIMS; i++)
	{
		diff_bits[i]=0;
		if( nbits == 1 )
		diff_bits[i] = __builtin_popcountll( coords[0][i] ^ coords[1][i]);
		else if( nbits == 2 )
		diff_bits[i] = __builtin_popcountll( (coords[0][i]^coords[1][i])|(coords[0][i]^coords[2][i])|(coords[0][i]^coords[3][i]) );

		diff_bits[i] += X/NUMDIMS;

		/*
		printf("org coord[%d] %llu \n", i, coord[i]);
		printCommonPrefix(coord[i] , 0 , 64, 64);
		printf("\n\n");
		*/

		coord[i] = LeftShift( RightShift( coord[i] , diff_bits[i]), diff_bits[i]);
		printf("min coord[%d] %llu \n", i, coord[i]);
		printCommonPrefix(coord[i] , 0 , 64, 64);
		printf("\n");
		rect[i] = coord[i]/1000000.0f;
		printf("min rect[%d] %f \n\n", i, rect[i]);

		coord[i] += (LeftShift( 1 , diff_bits[i] ) -1 );
		printf("max coord[%d] %llu \n", i, coord[i]);
		printCommonPrefix(coord[i] , 0 , 64, 64);
		printf("\n");
		rect[i+NUMDIMS] = coord[i]/1000000.0f;
		printf("max rect[%d] %f \n\n", i+NUMDIMS, rect[i+NUMDIMS]);
	}

}




int main()
{
	int NUMDIMS = 3;

	int i,j;
	bitmask_t coord[NUMDIMS];
	bitmask_t index = 2133213;

	printf("index %llu \n", index);
  printCommonPrefix(index , 0 , 64, 64);
	printf("\n\n");

	hilbert_i2c(NUMDIMS, 20, index, coord);
	printf("TTTTTTT1\n");
	for(i=0; i<NUMDIMS; i++)
		printf("%llu\n", coord[i]);

	int X = 1 ;
	printf("X : ");
	scanf("%d", &X);

	unsigned long long start_num = LeftShift( RightShift(index ,X), X);
	unsigned long long end_num = start_num | (LeftShift(1 , X) -1);

	printf("start num\n");
  printCommonPrefix(start_num , 0 , 64 , 64);
	printf("\n\n");
	printf("end num\n");
  printCommonPrefix(end_num , 0 , 64, 64);
	printf("\n\n");
	
	unsigned long long current_num;
	unsigned long long min[NUMDIMS];
	unsigned long long max[NUMDIMS];
	for(i=0; i<NUMDIMS; i++)
	{
	  min[i] = ULLONG_MAX;
	  max[i] = 0;
	}

	for(current_num = start_num; current_num <= end_num; current_num++)
	{
		hilbert_i2c(NUMDIMS, 20, current_num, coord);
	  for(i=0; i<NUMDIMS; i++)
		{
		  if( min[i] > coord[i] ) min[i] = coord[i];
		  if( max[i] < coord[i] ) max[i] = coord[i];
		  //printf("coord[%d] %llu\n",i, coord[i]);
		}
		if( current_num == ULLONG_MAX ) break;
	}

	for(i=0; i<NUMDIMS; i++)
	{
		printf("min[%d] %llu, ",i, min[i]);
		printCommonPrefix(min[i], 0 , NUMDIMS*10, 64);
		printf("\n");

		printf("max[%d] %llu, ",i, max[i]);
		printCommonPrefix(max[i], 0 , NUMDIMS*10, 64);
		printf("\n");
	}

	printf("Convert!!\n");
	float rect[NUMDIMS*2];
  ConvertHilbertIndexToBoundingBox(index, NUMDIMS, X, rect);

	/*
	printf("index %llu \n", index);
	for(i=0; i<NUMDIMS; i++)
	{
		printf("coord[%d] %llu\n", i, coord[i]);
    printCommonPrefix(coord[i] , 0 , NUMDIMS*10, 64);
  	printf("\n\n");
	}
	*/


	return 0;
}
