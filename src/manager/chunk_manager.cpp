#include "manager/chunk_manager.h"

#include "common/macro.h"

namespace ursus {
namespace manager {

/**
 * @brief Return the singleton chunk manager instance
 */
ChunkManager& ChunkManager::GetInstance(){
  static ChunkManager chunk_manager;
  return chunk_manager;
}

bool ChunkManager::Init(size_t size) {
  cudaErrCheck(cudaMalloc((void**) &d_node_soa_ptr, size));
  global_SetRootNode<<<1,1>>>(d_node_soa_ptr);
  cudaDeviceSynchronize();
  return true;
}

// Allocate in Pinned Memory
bool ChunkManager::InitInPinnedMemory(size_t size) {
  cudaErrCheck(cudaMallocHost((void**) &d_node_soa_ptr, size));
  global_SetRootNode<<<1,1>>>(d_node_soa_ptr);
  cudaDeviceSynchronize();
  return true;
}

/**
 * @brief copy the entire of partial nodes of the tree to the GPU
 * @return true if success otherwise false
 */ 
bool ChunkManager::CopyNode(node::Node_SOA* node_soa_ptr, ll offset, ui number_of_nodes) {
  cudaErrCheck(cudaMemcpy(&d_node_soa_ptr[offset], &node_soa_ptr[offset], 
               sizeof(node::Node_SOA)*number_of_nodes, cudaMemcpyHostToDevice));
  return true;
}

//===--------------------------------------------------------------------===//
// Cuda Variable & Function 
//===--------------------------------------------------------------------===//
__device__ node::Node_SOA* g_node_soa_ptr;

__global__ 
void global_SetRootNode(node::Node_SOA* d_node_soa_ptr) { 
  g_node_soa_ptr = d_node_soa_ptr;
}

} // End of manager namespace
} // End of ursus namespace

