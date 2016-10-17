#pragma once

#include "tree/tree.h"

#include <queue>
#include <mutex>

namespace ursus {
namespace tree {

class RTree_LS : public Tree {
 public:
  //===--------------------------------------------------------------------===//
  // Consteructor/Destructor
  //===--------------------------------------------------------------------===//
  RTree_LS();

 //===--------------------------------------------------------------------===//
 // Main Function
 //===--------------------------------------------------------------------===//

  /**
   * Build the RTree_LS tree with input_data_set
   */
  bool Build(std::shared_ptr<io::DataSet> input_data_set);

  bool DumpFromFile(std::string index_name);

  bool DumpToFile(std::string index_name);

  ui GetNumberOfNodeSOA() const;

  ui GetNumberOfLeafNodeSOA() const;

  ui GetNumberOfExtendLeafNodeSOA() const;

  ui GetChunkSize() const;

  /**
   * Search the data 
   */
  int Search(std::shared_ptr<io::DataSet> query_data_set, 
             ui number_of_search, ui number_of_repeat);

  void Thread_Search(std::vector<Point>&query, Point* d_query, 
                     ui tid, ui number_of_blocks_per_cpu, 
                     ui& node_visit_count, 
                     ui start_offset, ui end_offset) ;

  void SetChunkSize(ui chunk_size);

  void SetChunkUpdated(bool updated);

  void SetUpperTreeType(TreeType UPPER_TREE_TYPE);

  void SetNumberOfCPUThreads(ui number_of_cpu_threads);

  void SetNumberOfCUDABlocks(ui number_of_cuda_blocks);

  void RTree_LS_Search(node::Node *node_ptr, Point* query, Point* d_query,
                       ui query_offset, ui bid_offset, ui number_of_blocks_per_cpu, 
                       ui *node_visit_count);

  //===--------------------------------------------------------------------===//
  // Members
  //===--------------------------------------------------------------------===//
  // const : *ONLY* 'SetChunkSize' can modify the value 
  const ui chunk_size=4;

  bool chunk_updated=false;

  bool search_finish=false;

  bool upper_tree_exists=false;

  bool flat_array_exists=false;

  // basically, use single cpu thread
  const ui number_of_cpu_threads=1;
  
  std::vector<std::queue<ll>> thread_start_node_index;

  std::mutex chunk_updated_mutex;

  TreeType UPPER_TREE_TYPE;
};

//===--------------------------------------------------------------------===//
// Cuda Variable & Function 
//===--------------------------------------------------------------------===//

extern __device__ ui g_hit2[GetNumberOfMAXBlocks()]; 
extern __device__ ui g_node_visit_count2[GetNumberOfMAXBlocks()]; 
extern __device__ ui g_monitor[GetNumberOfMAXBlocks()]; 

__global__
void global_SetHitCount2(ui init_value);

__global__
void global_GetHitCount2(ui* hit, ui* node_visit_count);

__global__
void global_GetMonitor2(ui* monitor);

__global__ 
void global_RTree_LeafNode_Scan(Point* _query, ll start_node_offset, 
                                ui chunk_size, ui bid_offset,
                                ui number_of_blocks_per_cpu);
 
} // End of tree namespace
} // End of ursus namespace
