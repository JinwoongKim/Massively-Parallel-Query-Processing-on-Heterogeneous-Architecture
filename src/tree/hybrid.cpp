#include "tree/hybrid.h"

#include "common/macro.h"
#include "common/logger.h"
#include "evaluator/recorder.h"
#include "sort/thrust_sort.h"

#include <cassert>

namespace ursus {
namespace tree {

Hybrid::Hybrid() {
  tree_type = TREE_TYPE_HYBRID;
}

bool Hybrid::Build(std::shared_ptr<io::DataSet> input_data_set){
  LOG_INFO("Build Hybrid Tree");
  bool ret = false;

 //===--------------------------------------------------------------------===//
 // Create branches
 //===--------------------------------------------------------------------===//
  std::vector<node::Branch> branches = CreateBranches(input_data_set);

 //===--------------------------------------------------------------------===//
 // Assign Hilbert Ids to branches
 //===--------------------------------------------------------------------===//
  // TODO  have to choose policy later
  ret = AssignHilbertIndexToBranches(branches);
  assert(ret);

 //===--------------------------------------------------------------------===//
 // Sort the branches on the GPU
 //===--------------------------------------------------------------------===//
  ret = sort::Thrust_Sort::Sort(branches);
  assert(ret);

 //===--------------------------------------------------------------------===//
 // Build the internal nodes in a bottop-up fashion on the GPU
 //===--------------------------------------------------------------------===//
  ret = Bottom_Up(branches);
  assert(ret);

 //===--------------------------------------------------------------------===//
 // Transform the branches into SOA fashion 
 //===--------------------------------------------------------------------===//

 PrintTree();

  return true;
}

int Hybrid::Search(std::shared_ptr<io::DataSet> query_data_set){
  return -1  ;
}

bool Hybrid::Bottom_Up(std::vector<node::Branch> &branches) {
  auto& recorder = evaluator::Recorder::GetInstance();

 //===--------------------------------------------------------------------===//
 // Configure trees
 //===--------------------------------------------------------------------===//
  level_node_count = GetLevelNodeCount(branches);
  auto tree_height = level_node_count.size()-1;
  ui total_node_count = GetTotalNodeCount();
  auto leaf_node_offset = total_node_count-level_node_count[0];;

 //===--------------------------------------------------------------------===//
 // Copy the leaf nodes to the GPU
 //===--------------------------------------------------------------------===//
  node = new node::Node[total_node_count];
  // Copy the branches to nodes 
  auto ret = CopyToNode(branches, NODE_TYPE_LEAF, tree_height, leaf_node_offset);
  assert(ret);

  node::Node_Ptr d_node;
  cudaMalloc((void**) &d_node, sizeof(node::Node)*total_node_count);
  cudaMemcpy(d_node, node, sizeof(node::Node)*total_node_count, cudaMemcpyHostToDevice);
 //===--------------------------------------------------------------------===//
 // Construct the rest part of trees on the GPU
 //===--------------------------------------------------------------------===//
  recorder.TimeRecordStart();
  ul current_offset = total_node_count;
  for( ui range(level_itr, 0, tree_height)) {
    current_offset -= level_node_count[level_itr];
    ul parent_offset = (current_offset-level_node_count[level_itr+1]);
    BottomUpBuild_ILP(current_offset, parent_offset, level_node_count[level_itr], d_node);
  }
  // print out sorting time on the GPU
  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("BottomUp Construction Time on the GPU = %.6fs", elapsed_time/1000.0f);

  cudaMemcpy(node, d_node, sizeof(node::Node)*total_node_count, cudaMemcpyDeviceToHost);
  cudaFree(d_node);

  // Re-set child pointers
  SetChildPointers(node, total_node_count-level_node_count[0]);
  return true;
}

void Hybrid::PrintTree(ui count) {
  LOG_INFO("Height %zu", level_node_count.size());

  ui node_itr=0;

  for( int i=level_node_count.size()-1; i>=0; --i) {
    LOG_INFO("Level %zd", (level_node_count.size()-1)-i);
    for( ui range(j, 0, level_node_count[i])){
      LOG_INFO("node %p",&node[node_itr]);
      std::cout << node[node_itr++] << std::endl;

      if(count){ if( node_itr>=count){ return; } }
    }
  }

}

} // End of tree namespace
} // End of ursus namespace

