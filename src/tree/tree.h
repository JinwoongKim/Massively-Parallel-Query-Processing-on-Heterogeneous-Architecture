#pragma once

#include "common/types.h"
#include "io/dataset.h"
#include "node/node.h"

#include <memory>

namespace ursus {
namespace tree {

class Tree{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  Tree(){}

 //===--------------------------------------------------------------------===//
 // Virtual Function
 //===--------------------------------------------------------------------===//

  /**
   * Build the indexing structure
   */
  virtual bool Build(std::shared_ptr<io::DataSet> input_data_set) = 0;

  /**
   * Search the data 
   */
  virtual int Search(std::shared_ptr<io::DataSet> query_data_set) = 0;

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 protected:
  TreeType tree_type = TREE_TYPE_INVALID;

  unsigned int number_of_dimensions;

  node::Node root_node;
  
};

} // End of tree namespace
} // End of ursus namespace
