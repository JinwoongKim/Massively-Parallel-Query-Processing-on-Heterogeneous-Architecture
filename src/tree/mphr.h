#pragma once

#include "tree/tree.h"

namespace ursus {
namespace tree {

class MPHR : public Tree {
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  MPHR();

 //===--------------------------------------------------------------------===//
 // Main Function
 //===--------------------------------------------------------------------===//

  /**
   * Build the MPHRtrees with input_data_set
   */
  bool Build(std::shared_ptr<io::DataSet> input_data_set);

  /**
   * Search the data 
   */
  int Search(std::shared_ptr<io::DataSet> query_data_set);
};

} // End of tree namespace
} // End of ursus namespace
