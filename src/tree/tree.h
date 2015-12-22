#pragma once

#include "common/types.h"
#include "io/dataset.h"
#include "node/node.h"
#include "node/g_node.h"

#include <memory>
#include <vector>

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
   * Build the internal nodes
   */
  virtual bool Bottom_Up(std::vector<node::Branch> &branches) = 0;

  /**
   * Search the data 
   */
  virtual int Search(std::shared_ptr<io::DataSet> query_data_set) = 0;

 //===--------------------------------------------------------------------===//
 // Utility Function
 //===--------------------------------------------------------------------===//
  std::vector<node::Branch> CreateBranches(std::shared_ptr<io::DataSet> input_data_set) ;

  bool AssignHilbertIndexToBranches(std::vector<node::Branch> &branches);

  std::vector<ui> GetLevelNodeCount(std::vector<node::Branch> &branches);

  ui GetTotalNodeCount(void) const;

  bool CopyToNode(std::vector<node::Branch> &branches,
                  NodeType node_type, int level, ui offset);

  /**
   * Simple wrapper function
   */
  void BottomUpBuild_ILP(ul offset, ul parent_offset, ui number_of_node, 
                         node::Node_Ptr root);

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 protected:
  TreeType tree_type = TREE_TYPE_INVALID;

  // root node pointer
  node::Node_Ptr node;

  // number of nodes in each level
  std::vector<ui> level_node_count;

  // root node pointer on the GPU
  node::G_Node_Ptr g_node;

  // number of nodes in each level
  std::vector<ui> level_g_node_count;
  
};

} // End of tree namespace
} // End of ursus namespace
