#pragma once

#include "common/types.h"
#include "io/dataset.h"
#include "node/node.h"
#include "node/node_soa.h"

#include <memory>
#include <vector>

namespace ursus {
namespace tree {

class Tree {
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
  virtual bool Bottom_Up(std::vector<node::Branch> &branches);

  /**
   * Search the data 
   */
  virtual int Search(std::shared_ptr<io::DataSet> query_data_set, 
                     ui number_of_search) = 0;

  virtual void PrintTree(ui offset=0, ui count=0/*max*/);

  virtual void PrintTreeInSOA(ui offset=0, ui count=0/*max*/);

 //===--------------------------------------------------------------------===//
 // Utility Function
 //===--------------------------------------------------------------------===//
  std::vector<node::Branch> CreateBranches(std::shared_ptr<io::DataSet> input_data_set) ;

  void Thread_Mapping(std::vector<node::Branch> &branches, ui start_offset, ui end_offset);

  bool AssignHilbertIndexToBranches(std::vector<node::Branch> &branches);

  std::vector<ui> GetLevelNodeCount(const std::vector<node::Branch> &branches);

  ui GetTotalNodeCount(void) const;

  bool CopyBranchToNode(std::vector<node::Branch> &branches,
                        NodeType node_type, int level, ui offset);

  /**
   * wrapper function for Cuda 
   */
  void BottomUpBuild_ILP(ul offset, ul parent_offset, ui number_of_node, node::Node* root);

  void BottomUpBuildonCPU(ul current_offset, ul parent_offset, ui number_of_node, 
                         node::Node* root, ui tid, ui number_of_threads);

  bool MoveTreeToGPU(ui offset=0, ui count=0);

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 protected:
  node::Node* node_ptr;

  node::Node_SOA* node_soa_ptr;

  TreeType tree_type = TREE_TYPE_INVALID;

  // number of nodes in each level(start from leaf)
  std::vector<ui> level_node_count;

  // total node count 
  ui total_node_count;
};

//===--------------------------------------------------------------------===//
// Cuda function
//===--------------------------------------------------------------------===//
extern __device__ node::Node_SOA* g_node_soa_ptr;

__global__ 
void global_BottomUpBuild_ILP(ul current_offset, ul parent_offset,
                              ui number_of_node, node::Node* root);
__global__ 
void global_MoveTreeToGPU(node::Node_SOA* d_node_soa_ptr, ui total_node_count);

} // End of tree namespace
} // End of ursus namespace
