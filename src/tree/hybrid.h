#pragma once

#include "tree/tree.h"
#include "evaluator/recorder.h"

#include <queue>

namespace ursus {
namespace tree {

class Hybrid : public Tree {
 public:
  //===--------------------------------------------------------------------===//
  // Consteructor/Destructor
  //===--------------------------------------------------------------------===//
    Hybrid() = delete;
    Hybrid(ui number_of_cuda_blocks);

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

  void SetNumberOfNodeSOA(ui number_of_data);

  ui GetNumberOfNodeSOA() const;

  ui GetNumberOfLeafNodeSOA() const;

  ui GetNumberOfExtendLeafNodeSOA() const;

  /**
   * Search the data 
   */
  int Search(std::shared_ptr<io::DataSet> query_data_set, 
             ui number_of_search);

  void Thread_Search(std::vector<Point>&query, Point* d_query, 
                     ui tid, ui number_of_blocks_per_cpu, 
                     ui& jump_count, ui& node_visit_count, 
                     ui start_offset, ui end_offset) ;

  void SetChunkSize(ui chunk_size);

  // level to scan on the GPU
  // 1 : leaf nodes, 2 : extend and leaf nodes
  void SetScanType(ScanType scan_type);

  void SetNumberOfCPUThreads(ui number_of_cpu_threads);

  ll TraverseInternalNodes(node::Node *node_ptr, Point* query, 
                           ll passed_hIndex, ui *node_visit_count);

  ui BruteForceSearchOnCPU(Point* query);

  void Thread_BruteForce(Point* query, std::vector<ll> &start_node_offset, 
                         ui& hit, ui start_offset, ui end_offset);

  // Collect start node index in advance
  // to measure CPU/GPU execution time
  void Thread_CollectStartNodeIndex(std::vector<Point>& query,
                                    std::queue<ll> &start_node_indice,
                                    ui start_offset, ui end_offset);

  ll GetNextStartNodeIndex(ui tid);

  //===--------------------------------------------------------------------===//
  // Members
  //===--------------------------------------------------------------------===//
  ui chunk_size;

  ScanType scan_type;

  ui number_of_cpu_threads;
  
  ui leaf_node_soa_count = 0;

  ui extend_leaf_node_soa_count = 0;

  std::vector<std::queue<ll>> thread_start_node_index;
};

//===--------------------------------------------------------------------===//
// Cuda Variable & Function 
//===--------------------------------------------------------------------===//

extern __device__ ui g_hit[GetNumberOfMAXBlocks()]; 
extern __device__ ui g_node_visit_count[GetNumberOfMAXBlocks()]; 

__global__
void global_SetHitCount(ui init_value);

__global__
void global_GetHitCount(ui* hit, ui* node_visit_count);

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
