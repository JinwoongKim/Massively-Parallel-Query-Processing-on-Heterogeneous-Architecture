#pragma once

#include "tree/tree.h"
#include "evaluator/recorder.h"

namespace ursus {
namespace tree {

class MPHR : public Tree {
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  MPHR();
  ~MPHR();

 //===--------------------------------------------------------------------===//
 // Main Function
 //===--------------------------------------------------------------------===//

  /**
   * Build the MPHR tree with input_data_set
   */
  bool Build(std::shared_ptr<io::DataSet> input_data_set);

  bool DumpFromFile(std::string index_name);

  bool DumpToFile(std::string index_name);

  /**
   * Search the data 
   */
  int Search(std::shared_ptr<io::DataSet> query_data_set, 
             ui number_of_search, ui number_of_repeat);

  void SetNumberOfCUDABlocks(ui number_of_cuda_blocks);

  void SetNumberOfPartition(ui number_of_partition);

  //===--------------------------------------------------------------------===//
  // Members
  //===--------------------------------------------------------------------===//
  ui number_of_partition = 1;

  ll root_offset[GetNumberOfMAXBlocks()] = {0};
};

//===--------------------------------------------------------------------===//
// Cuda Function 
//===--------------------------------------------------------------------===//

extern __device__ ll g_root_offset[GetNumberOfMAXBlocks()];

__global__ 
void global_SetRootOffset(ll* root_offset);

__global__ 
void global_RestartScanning_and_ParentCheck(Point* query, ui* hit, 
                                 ui* root_visit_count, ui* node_visit_count);
 
} // End of tree namespace
} // End of ursus namespace
