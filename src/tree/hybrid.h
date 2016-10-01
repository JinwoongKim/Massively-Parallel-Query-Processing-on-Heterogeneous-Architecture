#pragma once

#include "tree/tree.h"

#include <queue>

namespace ursus {
namespace tree {

class Hybrid : public Tree {
 public:
  //===--------------------------------------------------------------------===//
  // Consteructor/Destructor
  //===--------------------------------------------------------------------===//
  Hybrid();

 //===--------------------------------------------------------------------===//
 // Main Function
 //===--------------------------------------------------------------------===//

  /**
   * Build the Hybrid tree with input_data_set
   */
  bool Build(std::shared_ptr<io::DataSet> input_data_set);

  bool DumpFromFile(std::string index_name);

  bool DumpToFile(std::string index_name);

  bool BuildExtendLeafNodeOnCPU();

  void Thread_BuildExtendLeafNodeOnCPU(ul current_offset, ul parent_offset, 
                                       ui number_of_node, ui tid, ui number_of_threads);

  ui GetNumberOfNodeSOA() const;

  ui GetNumberOfLeafNodeSOA() const;

  ui GetNumberOfExtendLeafNodeSOA() const;

  /**
   * Search the data 
   */
  int Search(std::shared_ptr<io::DataSet> query_data_set, 
             ui number_of_search, ui number_of_repeat);

  void Thread_Search(std::vector<Point>&query, Point* d_query, 
                     ui tid, ui number_of_blocks_per_cpu, 
                     ui& jump_count, ui& node_visit_count, 
                     ui start_offset, ui end_offset) ;

  void SetChunkSize(ui chunk_size);

  // level to scan on the GPU
  // 1 : leaf nodes, 2 : extend and leaf nodes
  void SetScanLevel(ui scan_level);

  void SetNumberOfCPUThreads(ui number_of_cpu_threads);

  void SetNumberOfCUDABlocks(ui number_of_cuda_blocks);

  ll TraverseInternalNodes(node::Node *node_ptr, Point* query, 
                           ll passed_hIndex, ui *node_visit_count);

  // Collect start node index in advance
  // to measure CPU/GPU execution time
  void Thread_CollectStartNodeIndex(std::vector<Point>& query,
                                    std::queue<ll> &start_node_indice,
                                    ui start_offset, ui end_offset);

  void Thread_Monitoring(ui update_interval);

  ll GetNextStartNodeIndex(ui tid);


  //===--------------------------------------------------------------------===//
  // Members
  //===--------------------------------------------------------------------===//
  // const : *ONLY* 'SetChunkSize' can modify the value 
  const ui chunk_size=128;

  bool chunk_updated=false;

  bool search_finish=false;

  const ui scan_level=1;
  // basically, use single cpu thread
  const ui number_of_cpu_threads=1;
  
  std::vector<ui> level_node_count;
  std::vector<std::queue<ll>> thread_start_node_index;
};

//===--------------------------------------------------------------------===//
// Cuda Variable & Function 
//===--------------------------------------------------------------------===//

extern __device__ ui g_hit[GetNumberOfMAXBlocks()]; 
extern __device__ ui g_node_visit_count[GetNumberOfMAXBlocks()]; 
extern __device__ ui g_monitor[GetNumberOfMAXBlocks()]; 

__global__
void global_SetHitCount(ui init_value);

__global__
void global_GetHitCount(ui* hit, ui* node_visit_count);

__global__
void global_GetMonitor(ui* monitor);

__global__ 
void global_ParallelScan_Leafnodes(Point* _query, ll start_node_offset, 
                                   ui chunk_size, ui bid_offset,
                                   ui number_of_blocks_per_cpu);
__global__ 
void global_ParallelScan_ExtendLeafnodes(Point* _query, ll start_node_offset, 
                                             ui chunk_size, ui bid_offset,
                                             ui number_of_blocks_per_cpu);
 
} // End of tree namespace
} // End of ursus namespace
