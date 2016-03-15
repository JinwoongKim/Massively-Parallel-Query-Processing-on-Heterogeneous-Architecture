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

  /**
   * Search the data 
   */
  int Search(std::shared_ptr<io::DataSet> query_data_set, 
             ui number_of_search);

  ull TraverseInternalNodes(node::Node *node_ptr, Point* query, 
                            ull passed_hIndex, ui *node_visit_count);

};

//===--------------------------------------------------------------------===//
// Cuda Function 
//===--------------------------------------------------------------------===//
__global__ 
void global_ParallelScanning_Leafnodes(Point* _query, ull start_node_offset, 
                                       ui chunk_size, ui* hit, ui* node_visit_count);
 
} // End of tree namespace
} // End of ursus namespace
