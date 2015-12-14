#pragma once

#include "common/types.h"
#include "io/dataset.h"
#include "node/node.h"

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
  virtual bool Build(io::DataSet* input_set) = 0;

  /**
   * Search the data 
   */
  virtual void Search(io::DataSet* query_set) = 0;

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 protected:
  TreeType tree_type = TREE_TYPE_INVALID;

  node::Node root_node;
  
};

} // End of tree namespace
} // End of ursus namespace
