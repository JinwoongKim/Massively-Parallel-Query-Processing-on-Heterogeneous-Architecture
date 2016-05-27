#pragma once

#include "common/types.h"
#include "node/node.h"
#include "node/node_soa.h"

namespace ursus {
namespace manager {

class ChunkManager {
  public:
  //===--------------------------------------------------------------------===//
  // Consteructor/Destructor
  //===--------------------------------------------------------------------===//
  ChunkManager(const ChunkManager &) = delete;
  ChunkManager &operator=(const ChunkManager &) = delete;
  ChunkManager(ChunkManager &&) = delete;
  ChunkManager &operator=(ChunkManager &&) = delete;

  // global singleton
  static ChunkManager& GetInstance(void);

  bool Init(size_t size);

  bool CopyNode(node::Node_SOA* node_soa_ptr, ll offset, ui number_of_nodes);

  //===--------------------------------------------------------------------===//
  // Members
  //===--------------------------------------------------------------------===//
  private:
  ChunkManager() {}
  node::Node_SOA* d_node_soa_ptr;

};

//===--------------------------------------------------------------------===//
// Cuda Variable & Function 
//===--------------------------------------------------------------------===//

extern __device__ node::Node_SOA* g_node_soa_ptr;

__global__ 
void global_SetRootNode(node::Node_SOA* d_node_soa_ptr);


} // End of manager namespace
} // End of ursus namespace
