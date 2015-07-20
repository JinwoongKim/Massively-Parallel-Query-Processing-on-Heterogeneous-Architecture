
#if !defined(_index_h_ )
#define _index_h_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>

#include <float.h>
#include <math.h>
#include "sys/time.h"
#include <sys/queue.h>
#include <queue>  // using queue C++ style
#include <list>  // using queue C++ style
#include <iostream>  // using queue C++ style
#include <cuda.h>
#include "hilbert.h"
#include "morton.h"

//Thrust library
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "cuda_profiler_api.h"

//for cuda checing
#include <assert.h>

//MPI
#include <mpi.h>
using namespace std;


#define NUMDIMS	3 // number of dimensions
#define DATATYPE "real"  // For NOAA real datasets, number of dimensions shoulde be 3
#define NODECARD 2     // number of nodecard , it MUST be square root of 2.

#define NUMTHREADS 64
#define NUM_TP 32

#define PGSIZE	(int) ( (sizeof(int)*2) + ( NODECARD*sizeof(struct Branch) ) ) // automatically calculate the pagesize 
#define BVH_PGSIZE	(int) ( (sizeof(int)*2) + ( NODECARD*sizeof(BVH_Branch) ) + 16 ) // automatically calculate the pagesize 

#endif


