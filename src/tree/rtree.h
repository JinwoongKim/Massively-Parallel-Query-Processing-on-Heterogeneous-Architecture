#pragma once

#include "tree/tree.h"

namespace ursus {
namespace tree {

class Rtree : public Tree {
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  Rtree();

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
             ui number_of_search, ui number_of_repeat);

  void Thread_Search(std::vector<Point>&query, 
                     ui tid, ui& hit, ui& node_visit_count, 
                     ui start_offset, ui end_offset) ;

  void SetNumberOfCPUThreads(ui number_of_cpu_threads);

  ui TraverseInternalNodes(node::Node *node_ptr, Point* query, 
                           ui *node_visit_count);

  //===--------------------------------------------------------------------===//
  // Members
  //===--------------------------------------------------------------------===//
  ui number_of_cpu_threads;
};

} // End of tree namespace
} // End of ursus namespace
