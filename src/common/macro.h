#define range_3(i,a,b)     i = (a); i < (b); ++i
#define range_4(i,a,b,j)   i = (a); i < (b); i+=j

#define range_X(x,i,a,b,j,FUNC, ...)  FUNC  

#define range(...)          range_X(,##__VA_ARGS__,\
                            range_4(__VA_ARGS__),\
                            range_3(__VA_ARGS__)\
                            ) 

#define ParallelReduction(array, size) \
        int N = size/2 + size%2; \
        while(N > 1) { \
          if( tid < N ) { \
            array[tid] = array[tid] + array[tid+N]; \
          } \
          N = N/2 + N%2; \
          __syncthreads(); \
        }\

#define FindLeftMostOverlappingChild(array, size) \
        int N = size/2 + size%2; \
        while(N > 1) { \
          if( tid < N ) { \
            if( array[tid] > array[tid+N]) { \
              array[tid] = array[tid+N]; \
            } \
          } \
          N = N/2 + N%2; \
          __syncthreads(); \
        }\
        if( tid == 0 ) { \
          if(N==1) { \
            if( array[0] > array[1]) { \
              array[0] = array[1]; \
            } \
          } \
        } \
        __syncthreads(); \

#define MasterThreadOnly\
        if( tid == 0 )\

#include <iostream>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
