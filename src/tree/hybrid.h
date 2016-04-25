#pragma once

#include "tree/tree.h"
#include "evaluator/recorder.h"

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

  /**
   * Search the data 
   */
  int Search(std::shared_ptr<io::DataSet> query_data_set, 
             ui number_of_search);

  void SetChunkSize(ui chunk_size);

  ll TraverseInternalNodes(node::Node *node_ptr, Point* query, 
                           ll passed_hIndex, ui *node_visit_count);

  bool BruteForceSearchOnCPU(Point* query);

  void Thread_BruteForce(Point* query, std::vector<ll> &start_node_offset, 
                         ui& hit, ui start_offset, ui end_offset);

  //===--------------------------------------------------------------------===//
  // Members
  //===--------------------------------------------------------------------===//
  ui chunk_size;
};

//===--------------------------------------------------------------------===//
// Cuda Variable & Function 
//===--------------------------------------------------------------------===//

extern __device__ ui g_hit[GetNumberOfBlocks()]; 
extern __device__ ui g_node_visit_count[GetNumberOfBlocks()]; 

__global__
void global_SetHitCount(ui init_value);

__global__
void global_GetHitCount(ui* hit, ui* node_visit_count);

__global__ 
void global_ParallelScanning_Leafnodes(Point* _query, ll start_node_offset, 
                                       ui chunk_size);
 
} // End of tree namespace
} // End of ursus namespace
