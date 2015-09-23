#include <morton.h>

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits(unsigned int v)
{
	//	printf(" before expand v : %d\n", v);
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;

	//	printf(" after expand v : %d\n", v);
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
/*
	unsigned int morton3D(float x, float y, float z)
	{
//	printf("x %.6f y %.6f z %.6f\n",x,y,z);
x = min(max(x * 1024.0f, 0.0f), 1023.0f);
y = min(max(y * 1024.0f, 0.0f), 1023.0f);
z = min(max(z * 1024.0f, 0.0f), 1023.0f);
//	printf(" min max x : %.6f\n", x);
//	printf(" min max y : %.6f\n", y);
//	printf(" min max z : %.6f\n", z);
unsigned int xx = expandBits((unsigned int)x);
unsigned int yy = expandBits((unsigned int)y);
unsigned int zz = expandBits((unsigned int)z);
return xx * 4 + yy * 2 + zz;
}
 */

// method to seperate bits from a given integer 3 positions apart
inline unsigned long long splitBy3(unsigned long long a){
	unsigned long long x = a & 0x1fffff; // we only look at the first 21 bits

	x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
	x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
	x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
	x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
	x = (x | x << 2) & 0x1249249249249249;

	return x;
}

// method to seperate bits from a given integer 3 positions apart
inline unsigned long long splitByDim(unsigned long long a, const unsigned dim){

	//	unsigned long long x = a & 0x1fffff; // we only look at the first 21 bits       00000000000000000000000000000000000111111111111111111111
	unsigned long long x;
	unsigned int shift = (64/dim); //valid digits
	x = a << shift ; //remove unnecessary digits
	x = x >> shift ;

	unsigned long long valid_bit[shift];
	for( int i=0 ; i < shift ; i++)
		if( (x>>i)%2 == 0 )
			valid_bit[i] = 0;
		else
			valid_bit[i] = 1;

	x &= 0; 

	for( int i=0 ; i < shift ; i++)
		x |= ( valid_bit[i] << (i*dim) );

	return x;
}

inline unsigned long long mortonEncode_magicbits(unsigned long long x, unsigned long long y, unsigned long long z){
	unsigned long long answer = 0;
	//answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
	answer = answer | splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
	return answer;
}
inline unsigned long long mortonEncode_magicbits2(unsigned long long *input, const unsigned int dim)
{
	unsigned long long answer = 0;

	for(int i=0 ; i < dim; i++)
		answer |= splitByDim(input[i], dim) << i;

	return answer;
}

