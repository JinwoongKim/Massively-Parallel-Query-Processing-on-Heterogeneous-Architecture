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
 // Virtual Function
 //===--------------------------------------------------------------------===//
  /**
   * Build the indexing structure
   */
  virtual bool Build(std::shared_ptr<io::DataSet> input_data_set) =0;

  virtual bool DumpFromFile(std::string index_name) =0;

  virtual bool DumpToFile(std::string index_name) =0;

  /**
   * Build the internal nodes
   */
  bool Top_Down(std::vector<node::Branch> &branches);

  bool RTree_Top_Down(std::vector<node::Branch> &branches);

  bool Bottom_Up(std::vector<node::Branch> &branches);

  /**
   * Search the data 
   */
  virtual int Search(std::shared_ptr<io::DataSet> query_data_set, 
                     ui number_of_search, ui number_of_repeat) =0;

  void PrintTree(ui offset, ui count);

  void PrintTreeInSOA(ui offset, ui count);

 //===--------------------------------------------------------------------===//
 // Accessor
 //===--------------------------------------------------------------------===//
  TreeType GetTreeType() const;

  std::string GetIndexName(std::shared_ptr<io::DataSet> input_data_set);

  FILE* OpenIndexFile(std::string index_name);

 //===--------------------------------------------------------------------===//
 // Utility Function
 //===--------------------------------------------------------------------===//
  std::vector<node::Branch> CreateBranches(std::shared_ptr<io::DataSet> input_data_set) ;

  node::Node* CreateNode(std::vector<node::Branch> &branches, 
                         ui start_offset, ui end_offset, int level,
                         std::vector<ui>& level_node_count);

  ui GetSplitOffset(std::vector<node::Branch> &branches,
                    ui start_offset, ui end_offset);

  std::vector<ui> GetSplitPosition(std::vector<node::Branch> &branches, 
                                   ui start_offset, ui end_offset);

  void Thread_SetRect(std::vector<node::Branch> &branches, std::vector<Point>& points, 
                      ui start_offset, ui end_offset);


  void Thread_Mapping(std::vector<node::Branch> &branches, ui start_offset, ui end_offset);

  bool AssignHilbertIndexToBranches(std::vector<node::Branch> &branches);

  bool ClusterBrancheUsingKmeans(std::vector<node::Branch> &branches);

  std::vector<ui> GetLevelNodeCount(const std::vector<node::Branch> branches);

  ui GetDeviceNodeCount(const std::vector<ui> level_node_count);

  ui GetNumberOfBlocks(void) const;

  void Thread_CopyBranchToNode(std::vector<node::Branch> &branches, 
                               node::Node* node_ptr, NodeType node_type,
                               int level, ui node_offset, 
                               ui start_offset, ui end_offset);

  void Thread_CopyBranchToNodeSOA(std::vector<node::Branch> &branches, 
                               NodeType node_type,int level, ui node_offset, 
                               ui start_offset, ui end_offset);

  bool CopyBranchToNode(std::vector<node::Branch> &branches,
                        NodeType node_type, int level, ui node_offset,
                        node::Node* node_ptr);

  bool CopyBranchToNodeSOA(std::vector<node::Branch> &branches, 
                           NodeType node_type,int level, ui node_offset);

  ui BruteForceSearchOnCPU(Point* query);

  void Thread_BruteForce(Point* query, std::vector<ll> &start_node_offset, 
                         ui& hit, ui start_offset, ui end_offset);

  void Thread_BruteForceInSOA(Point* query, std::vector<ll> &start_node_offset,
                             ui &hit, ui start_offset, ui end_offset);

    /**
   * wrapper function for Cuda 
   */
  void BottomUpBuild_ILP(ul offset, ul parent_offset, ui number_of_node, node::Node* root);

  void BottomUpBuildonCPU(ul current_offset, ul parent_offset, ui number_of_node, 
                         node::Node* root, ui tid, ui number_of_threads);

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 protected:
  // # of cuda blocks
  ui number_of_cuda_blocks = 0;

  node::Node* node_ptr = nullptr;

  // TODO tmp node ptr for bottom up construnction
  node::Node* b_node_ptr = nullptr;

  node::Node_SOA* node_soa_ptr = nullptr;

  TreeType tree_type = TREE_TYPE_INVALID;

  // For BVH and Hybrid trees
  ui host_node_count = 0;

  // For MPHR and Hybrid Trees
  ui device_node_count = 0;
};

//===--------------------------------------------------------------------===//
// Cuda function
//===--------------------------------------------------------------------===//
__global__ 
void global_BottomUpBuild_ILP(ul current_offset, ul parent_offset,
                              ui number_of_node, node::Node* root,
                              ui number_of_cuda_blocks);
} // End of tree namespace
} // End of ursus namespace
