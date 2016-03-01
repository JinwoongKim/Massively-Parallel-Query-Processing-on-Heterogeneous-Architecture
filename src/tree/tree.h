#pragma once

#include "common/types.h"
#include "io/dataset.h"
#include "node/node.h"
#include "node/node_soa.h"

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

  /**
   * Print tree
   */
  virtual void PrintTree(ui count=0) = 0;

 //===--------------------------------------------------------------------===//
 // Utility Function
 //===--------------------------------------------------------------------===//
  std::vector<node::Branch> CreateBranches(std::shared_ptr<io::DataSet> input_data_set) ;

  bool AssignHilbertIndexToBranches(std::vector<node::Branch> &branches);

  std::vector<ui> GetLevelNodeCount(std::vector<node::Branch> &branches);

  ui GetTotalNodeCount(void) const;

  bool CopyToNode(std::vector<node::Branch> &branches,
                  NodeType node_type, int level, ui offset);

  void SetChildPointers(node::Node_Ptr node, ui number_of_nodes);
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
  node::Node* node_ptr;

  // root pointer for node_soa
  node::Node_SOA* node_soa_ptr;

  // number of nodes in each level
  std::vector<ui> level_node_count;

  // total node count 
  ui total_node_count;
  
};

// I don't know how to make it a member variable
// root pointer for node_soa on the GPU
// TODO
//__device__ node::Node_SOA* g_node_soa;

} // End of tree namespace
} // End of ursus namespace
