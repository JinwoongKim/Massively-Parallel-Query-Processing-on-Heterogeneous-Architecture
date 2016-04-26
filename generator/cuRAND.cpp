/*
 * This program uses the host CURAND API to generate pseudorandom floats. 
 */ 

#include <stdio.h> 
#include <stdlib.h> 
#include <cuda.h> 
#include <curand.h> 

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__);\
  return EXIT_FAILURE;}} while(0) 
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__); \
  return EXIT_FAILURE;}} while(0) 

int main(int argc, char *argv[]) { 

  if(argc<3){
    printf("Usage: %s # data # dims (opt : offset)\n", argv[0]);
    exit(0);
  }

  int nData = atoi(argv[1]); 
  int nDims = atoi(argv[2]); 
  int offset = 0;
  if( argc==4) {
    offset = atoi(argv[3]); 
    offset *= nDims;
  }

  char filename[100];
  sprintf(filename, "synthetic_%dd_%d_data.bin", nDims, nData);
  FILE *fp = fopen(filename, "w");

  curandGenerator_t gen; 
  float *devData, *hostData; 

  /* Allocate floats on host */
  hostData = (float *)calloc(nData*nDims, sizeof(float)); 

  /* Allocate floats on device */ 
  CUDA_CALL(cudaMalloc((void **)&devData, nData*nDims*sizeof(float))); 

  /* Create pseudo-random number generator */ 
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); 

  /* Set seed */ 
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL)); 

  CURAND_CALL(curandSetGeneratorOffset(gen, offset));

  /* Generate floats on device */ 
  CURAND_CALL(curandGenerateUniform(gen, devData, nData*nDims)); 

  /* Copy device memory to host */ 
  CUDA_CALL(cudaMemcpy(hostData, devData, nData*nDims*sizeof(float), cudaMemcpyDeviceToHost)); 

  /* Show result */ 
  for(int i=1; i<=nData*nDims; i++) { 
    //printf("%1.4f ", hostData[i]); 
    fwrite(&hostData[i], sizeof(float), 1, fp);
    //if( i%nDims == 0 ) printf("\n");
  } 
  printf("\n"); 


  /* Cleanup */ 
  CURAND_CALL(curandDestroyGenerator(gen)); 
  CUDA_CALL(cudaFree(devData)); 
  free(hostData); 
  fclose(fp);

  return EXIT_SUCCESS; 
} 

/*

NOTE : Distribution Guide

The curandGenerateUniform() function is used to generate uniformly distributed floating point values between 0.0 and 1.0, where 0.0 is excluded and 1.0 is included.

curandStatus_t 
curandGenerateNormal( curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev)
The curandGenerateNormal() function is used to generate normally distributed floating point values with the given mean and standard deviation.

curandStatus_t 
curandGenerateLogNormal( curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev)
The curandGenerateLogNormal() function is used to generate log-normally distributed floating point values based on a normal distribution with the given mean and standard deviation.

curandStatus_t 
curandGeneratePoisson( curandGenerator_t generator, unsigned int *outputPtr, size_t n, double lambda)
The curandGeneratePoisson() function is used to generate Poisson-distributed integer values based on a Poisson distribution with the given lambda.

curandStatus_t
curandGenerateUniformDouble( curandGenerator_t generator, double *outputPtr, size_t num)
The curandGenerateUniformDouble() function generates uniformly distributed random numbers in double precision.

curandStatus_t
curandGenerateNormalDouble( curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev)
curandGenerateNormalDouble() generates normally distributed results in double precision with the given mean and standard deviation. Double precision results can only be generated on devices of compute capability 1.3 or above, and the host.

curandStatus_t
curandGenerateLogNormalDouble( curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev)
curandGenerateLogNormalDouble() generates log-normally distributed results in double precision, based on a normal distribution with the given mean and standard deviation.

Read more at: http://docs.nvidia.com/cuda/curand/index.html#ixzz46p7V1uPc 
Follow us: @GPUComputing on Twitter | NVIDIA on Facebook

 */
