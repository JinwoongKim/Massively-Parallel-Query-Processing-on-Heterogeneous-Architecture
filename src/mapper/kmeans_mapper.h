#pragma once

#include "common/types.h"
#include "common/config.h"
#include "node/branch.h"

namespace ursus {
namespace mapper {

class KmeansMapper {
 public:
 static bool ClusteringBranches(std::vector<node::Branch> &branches, 
                                const ui number_of_dims);

  __both__ static 
  Point euclid_dist_2(int numCoords, int numObjs, int numClusters, Point *objects,
                      Point *clusters, int objectId, int clusterId);

 private:
  static std::vector<node::Branch> cuda_kmeans(const std::vector<node::Branch> &objects,   
                                               const int number_of_dims, 
                                               const int number_of_clusters, 
                                               int *clusterIDs);
};

__global__ 
void find_nearest_cluster(int numCoords, int numObjs, int numClusters,
                          Point *objects, Point *deviceClusters, int *membership,
                          int *intermediates, int offset);
__global__ 
void compute_delta(int *deviceIntermediates, int numIntermediates, int numThreads);

} // End of mapper namespace
} // End of ursus namespace
