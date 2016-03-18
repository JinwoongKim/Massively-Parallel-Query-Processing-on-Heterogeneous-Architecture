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
