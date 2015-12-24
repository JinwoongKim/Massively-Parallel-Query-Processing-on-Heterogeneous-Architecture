#pragma once

#include "tree/tree.h"

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
  int Search(std::shared_ptr<io::DataSet> query_data_set);

  /**
   * Build the internal nodes
   */
  bool Bottom_Up(std::vector<node::Branch> &branches);

 //===--------------------------------------------------------------------===//
 // Utility
 //===--------------------------------------------------------------------===//
  void PrintTree(ui count=0);
};

} // End of tree namespace
} // End of ursus namespace
