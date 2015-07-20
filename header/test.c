#include <stdio.h>
#include "hilbert.h"

//hilbert_c2i(unsigned nDims, unsigned nBits, bitmask_t const coord[])

int main()
{
	int DIMS = 2;
	long long input[4][DIMS];

	long long order2[16][DIMS];

	order2[0][0] = 0;
	order2[0][1] = 0;

	order2[1][0] = 0;
	order2[1][1] = 1;

	order2[2][0] = 1;
	order2[2][1] = 1;

	order2[3][0] = 1;
	order2[3][1] = 0;

	order2[4][0] = 2;
	order2[4][1] = 0;

	order2[5][0] = 3;
	order2[5][1] = 0;

	order2[6][0] = 3;
	order2[6][1] = 1;

	order2[7][0] = 2;
	order2[7][1] = 1;

	order2[8][0] = 2;
	order2[8][1] = 2;

	order2[9][0] = 3;
	order2[9][1] = 2;

	order2[10][0] = 3;
	order2[10][1] = 3;

	order2[11][0] = 2;
	order2[11][1] = 3;

	order2[12][0] = 1;
	order2[12][1] = 3;

	order2[13][0] = 1;
	order2[13][1] = 2;
	
	order2[14][0] = 0;
	order2[14][1] = 2;

	order2[15][0] = 0;
	order2[15][1] = 3;


	input[0][0]=0;
	input[0][1]=0;

	input[1][0]=1;
	input[1][1]=0;

	input[2][0]=0;
	input[2][1]=1;

	input[3][0]=1;
	input[3][1]=1;

	long long result;
	int i;

	for(i=0; i<4; i++)
	{
		result = hilbert_c2i(DIMS, 2, input[i]); 
		printf("%lu\n",result);
	}

	for(i=0; i<4; i++)
	{
		result = hilbert_c2i(DIMS, 4, input[i]); 
		printf("%lu\n",result);
	}

	for(i=0; i<4; i++)
	{
		result = hilbert_c2i(DIMS, 8, input[i]); 
		printf("%lu\n",result);
	}
	printf("order 2\n");

	for(i=0; i<16; i++)
	{
		result = hilbert_c2i(DIMS, 4, order2[i]); 
		printf("%lu\n",result);
	}

	long long test[2]={0,15};

	result = hilbert_c2i(DIMS, 8, test); 
	printf("%lu\n",result);




	return 0;
}


