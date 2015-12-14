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
   * Build the indexing structure
   */
  bool Build(io::DataSet *input_data_set);

  /**
   * Search the data 
   */
  void Search(io::DataSet *query_data_set);

};

} // End of tree namespace
} // End of ursus namespace
