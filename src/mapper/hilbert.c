/* C header file for Hilbert curve functions */
#include "hilbert.h"

#ifdef __cplusplus
extern "C" {
#endif

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
				coords ^= coords >> b;
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
			//printf("before bitTranspose %lu\n", coords);
			coords = bitTranspose(nDims, nBits, coords);
			//printf("after bitTranspose %lu\n", coords);
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

