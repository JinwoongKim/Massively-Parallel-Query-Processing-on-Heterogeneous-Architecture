#include "hilbert_mapper.h"

#include "common/macro.h"
#include "common/logger.h"
#include "sort/sorter.h"
#include "kmeans_macro.h"
#include "kmeans_mapper.h"

#include <vector>

namespace ursus {
namespace mapper {

/* return an array of cluster centers of size [number_of_clusters][number_of_dims]       */
std::vector<node::Branch> 
KmeansMapper::cuda_kmeans(const std::vector<node::Branch> &objects,   
                          const int     number_of_dims, 
                          const int     number_of_clusters, 
                          int    *clusterIDs) {
  int   i, j, index, loop=0;
  int   *newClusterSize; /* [number_of_clusters]: no. objects assigned in each
                              new cluster */
  Point    delta;          /* % of objects change their clusters */
  Point  **dimObjects;
  Point  **dimClusters;
  Point  **newClusters;    /* [number_of_dims][number_of_clusters] */

  Point *deviceClusters;
  Point *deviceObjects;
  int *deviceMembership;
  int *deviceIntermediates;
  int *tempIndexes;

  int numObjs = objects.size();


  //  Copy objects given in [numObjs][number_of_dims] layout to new
  //  [number_of_dims][numObjs] layout
  malloc2D(dimObjects, number_of_dims, numObjs, Point);
  for (i = 0; i < number_of_dims; i++) {
    for (j = 0; j < numObjs; j++) {
      dimObjects[i][j] = objects[j].GetPoint(i);
    }
  }

  /* pick first number_of_clusters elements of objects[] as initial cluster centers*/
  malloc2D(dimClusters, number_of_dims, number_of_clusters, Point);
  // MH: I Think this is bad choice, how about random?
  /*
     for (i = 0; i < number_of_dims; i++) {
     for (j = 0; j < number_of_clusters; j++) {
     dimClusters[i][j] = dimObjects[i][j];
     }
     }
   */
  //srand(time(NULL));
  srand(0);
  tempIndexes = (int*)malloc(number_of_clusters * sizeof(int));
  for (i = 0; i < number_of_clusters; i++) {
    int tmp = rand() % numObjs;
    for (j = 0; j < i; j++) {
      if (tmp == tempIndexes[j]) {
        tmp = rand() % numObjs;
        j = 0;
      }
    }
    tempIndexes[i] = tmp;
  }

  for (i = 0; i < number_of_dims; i++) {
    for (j = 0; j < number_of_clusters; j++) {
      dimClusters[i][j] = dimObjects[i][tempIndexes[j]];
    }
  }

  /* initialize clusterIDs[] */
  for (i=0; i<numObjs; i++) clusterIDs[i] = -1;

  /* need to initialize newClusterSize and newClusters[0] to all 0 */
  newClusterSize = (int*) calloc(number_of_clusters, sizeof(int));
  assert(newClusterSize != NULL);

  malloc2D(newClusters, number_of_dims, number_of_clusters, Point);
  memset(newClusters[0], 0, number_of_dims * number_of_clusters * sizeof(Point));

  //  To support reduction, numThreadsPerClusterBlock *must* be a power of
  //  two, and it *must* be no larger than the number of bits that will
  //  fit into an unsigned char, the type used to keep track of clusterIDs
  //  changes in the kernel.
  const unsigned int numThreadsPerClusterBlock = 128;
  const unsigned int numClusterBlocks =
    (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
  const unsigned int clusterBlockSharedDataSize =
    numThreadsPerClusterBlock * sizeof(unsigned char);

  const unsigned int numReductionThreads = nextPowerOfTwo(numClusterBlocks);
  cudaErrCheck(cudaMalloc(&deviceObjects, numObjs*number_of_dims*sizeof(Point)));
  cudaErrCheck(cudaMalloc(&deviceClusters, number_of_clusters*number_of_dims*sizeof(Point)));
  cudaErrCheck(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
  cudaErrCheck(cudaMalloc(&deviceIntermediates, numReductionThreads*sizeof(unsigned int)));

  cudaErrCheck(cudaMemcpy(deviceObjects, dimObjects[0],
        numObjs*number_of_dims*sizeof(Point), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(deviceMembership, clusterIDs,
        numObjs*sizeof(int), cudaMemcpyHostToDevice));

  do {
    cudaErrCheck(cudaMemcpy(deviceClusters, dimClusters[0],
          number_of_clusters*number_of_dims*sizeof(Point), cudaMemcpyHostToDevice));

    // DEBUG::::numClusterBlocks is THE PROBLEM
    //		printf("numClusterBlocks = %d, numThreadsPerClusterBlock = %d, clusterBlockSharedDataSize = %d\n",
    //			numClusterBlocks, numThreadsPerClusterBlock,clusterBlockSharedDataSize);
    if ( numClusterBlocks > 62500 ) {
      for ( int offset = 0; offset < numClusterBlocks; offset += 62500 ) {
        find_nearest_cluster
          <<< 62500, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
          (number_of_dims, numObjs, number_of_clusters,
           deviceObjects, deviceClusters, deviceMembership, deviceIntermediates, offset);
      }
    } else {
      find_nearest_cluster
        <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
        (number_of_dims, numObjs, number_of_clusters,
         deviceObjects, deviceClusters, deviceMembership, deviceIntermediates, 0);
    }

    cudaDeviceSynchronize(); 

    compute_delta <<< 1, 1024, 1024 * sizeof(int) >>>
      (deviceIntermediates, numClusterBlocks, 1024);
    //printf("compute delta executed\n");
    //printf("numReductionThreads=%d, reductionBlockSharedDataSize=%d\n", numReductionThreads, reductionBlockSharedDataSize);

    cudaDeviceSynchronize(); 

    int d;
    cudaErrCheck(cudaMemcpy(&d, deviceIntermediates,
          sizeof(int), cudaMemcpyDeviceToHost));
    delta = (Point)d;

    cudaErrCheck(cudaMemcpy(clusterIDs, deviceMembership,
          numObjs*sizeof(int), cudaMemcpyDeviceToHost));

    for (i=0; i<numObjs; i++) {
      /* find the array index of nestest cluster center */
      index = clusterIDs[i];

      /* update new cluster centers : sum of objects located within */
      newClusterSize[index]++;
      for (j=0; j<number_of_dims; j++)
        newClusters[j][index] += objects[i].GetPoint(j);
    }

    //  TODO: Flip the nesting order
    //  TODO: Change layout of newClusters to [number_of_clusters][number_of_dims]
    /* average the sum and replace old cluster centers with newClusters */
    for (i=0; i<number_of_clusters; i++) {
      for (j=0; j<number_of_dims; j++) {
        if (newClusterSize[i] > 0)
          dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
        newClusters[j][i] = 0.0;   /* set back to 0 */
      }
      newClusterSize[i] = 0;   /* set back to 0 */
    }

    delta /= numObjs;
    LOG_INFO("Loop : %d", loop);
    //printf("%*d - delta = %.4f\n", 4, loop, delta);
  } while (delta > DFN_KmeansThreashhold && loop++ < kMeansLoopIteration);

  printf("cudaKNNSearch(): %d iteartions\n", loop+1);

  /* allocate a 2D space for returning variable clusters[] (coordinates
     of cluster centers) */
  //malloc2D(clusters, number_of_clusters, number_of_dims, Point);
  std::vector<node::Branch> clusters;
  clusters.resize(number_of_clusters);

  for (i = 0; i < number_of_clusters; i++) {
    for (j = 0; j < number_of_dims; j++) {
      clusters[i].SetPoint(dimClusters[j][i],j);
    }
  }

  cudaErrCheck(cudaFree(deviceObjects));
  cudaErrCheck(cudaFree(deviceClusters));
  cudaErrCheck(cudaFree(deviceMembership));
  cudaErrCheck(cudaFree(deviceIntermediates));

  free(dimObjects[0]);
  free(dimObjects);
  free(dimClusters[0]);
  free(dimClusters);
  free(newClusters[0]);
  free(newClusters);
  free(newClusterSize);
  free(tempIndexes);

  return clusters;
}


bool KmeansMapper::ClusteringBranches(std::vector<node::Branch> &branches, 
                                      const ui number_of_dims){

  int* clusterIDs = new int[branches.size()]; // cluster id for each data 

  std::vector<node::Branch> clusters = cuda_kmeans(branches, number_of_dims, 
                                                   DFN_NumberOfClusters, 
                                                   clusterIDs);

  std::vector<Point> points;
  points.resize(number_of_dims);

  for ( int i = 0; i < DFN_NumberOfClusters; i++ ) {
    for ( int dim = 0; dim < number_of_dims; dim++ ) {
      points[dim] = clusters[i].GetPoint(dim);
    }

    ui number_of_bits = (number_of_dims>2) ? 20:31;
    clusters[i].SetIndex(HilbertMapper::MappingIntoSingle(number_of_dims, number_of_bits, points ));
    clusters[i].SetChildOffset(i); // keep cluters' order
  }

  // Sort the clusters
  bool ret;
  ret = sort::Sorter::Sort(clusters);
  assert(ret);

  // swap the index and child offset
  for (auto& cluster : clusters) {
    auto index = cluster.GetIndex();
    auto position = cluster.GetChildOffset();
    cluster.SetIndex(position);
    cluster.SetChildOffset(index);
  }

  // Sort the clusters
  ret = sort::Sorter::Sort(clusters);
  assert(ret);

  // swap the index and child offset 
  for (auto& cluster : clusters) {
    auto index = cluster.GetIndex();
    auto position = cluster.GetChildOffset();
    cluster.SetIndex(position);
    cluster.SetChildOffset(index);
  }

  for ( int i = 0; i < branches.size()-1; i++ ) {
    ll index = clusters[clusterIDs[i]].GetIndex()*branches.size() + branches[i].GetIndex();
    branches[i].SetIndex(index);
  }

  delete clusterIDs;

  return true;
}

/* square of Euclid distance between two multi-dimensional points */
__both__ 
Point KmeansMapper::euclid_dist_2(int    numCoords,
                                  int    numObjs,
                                  int    numClusters,
                                  Point *objects,     // [numCoords][numObjs]
                                  Point *clusters,    // [numCoords][numClusters]
                                  int    objectId,
                                  int    clusterId) {
  Point ans=0.0;

  for(ui range(i, 0, numCoords)) {
    ans += (objects[numObjs*i+objectId]-clusters[numClusters*i+clusterId])*
           (objects[numObjs*i+ objectId]-clusters[numClusters*i+clusterId]);
  }

  return(ans);
}

__global__ 
void find_nearest_cluster(int numCoords, int numObjs, int numClusters,
                                        Point *objects, //[numCoords][numObjs]
                                        Point *deviceClusters,//[numCoords][numClusters]
                                        int *clusterIDs, //[numObjs]
                                        int *intermediates, int offset) {
    extern __shared__ char sharedMemory[];

    //  The type chosen for membershipChanged must be large enough to support
    //  reductions! There are blockDim.x elements, one for each thread in the
    //  block. See numThreadsPerClusterBlock in cuda_kmeans().
    unsigned char *membershipChanged = (unsigned char *)sharedMemory;
    Point *clusters = deviceClusters;

    membershipChanged[threadIdx.x] = 0;

    int objectId = blockDim.x * (blockIdx.x + offset) + threadIdx.x;

    if (objectId < numObjs) {
        int   index, i;
        Point dist, min_dist;

        /* find the cluster id that has min distance to object */
        index    = 0;
        min_dist = KmeansMapper::euclid_dist_2(numCoords, numObjs, numClusters,
                                               objects, clusters, objectId, 0);

        for (i=1; i<numClusters; i++) {
            dist = KmeansMapper::euclid_dist_2(numCoords, numObjs, numClusters,
                                               objects, clusters, objectId, i);
            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        if (clusterIDs[objectId] != index) {
            membershipChanged[threadIdx.x] = 1;
        }

        /* assign the clusterIDs to object objectId */
        clusterIDs[objectId] = index;

        __syncthreads();    //  For membershipChanged[]

        //  blockDim.x *must* be a power of two!
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                membershipChanged[threadIdx.x] +=
                    membershipChanged[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            intermediates[blockIdx.x + offset] = membershipChanged[0];
        }
    }
}

__global__ 
void compute_delta(int *deviceIntermediates, int numIntermediates,
				                         int numThreads) {

	extern __shared__ unsigned int intermediates[];
	const int tid = threadIdx.x;

	intermediates[tid] = 0;
	__syncthreads();
	for (int i = 0; i < numIntermediates; i+=numThreads)
		intermediates[tid] += deviceIntermediates[i];
	__syncthreads();

	int N = numThreads/2 + numThreads%2;
	while(N > 1) {
		if (tid < N)
			intermediates[tid] += intermediates[tid+N];
		__syncthreads();
		N = N/2 + N%2;
	}

	if ( tid == 0 ) {
		if ( N == 1 ) {
			intermediates[0] += intermediates[1];
		}
	}
	__syncthreads();

	deviceIntermediates[0] = intermediates[0];
}

} // End of mapper namespace
} // End of ursus namespace
