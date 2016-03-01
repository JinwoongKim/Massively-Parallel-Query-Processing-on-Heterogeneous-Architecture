#include "tree/tree.h"

#include "common/macro.h"
#include "mapper/hilbert_mapper.h"

#include <cmath>

namespace ursus {
namespace tree {

//===--------------------------------------------------------------------===//
// Cuda Function
//===--------------------------------------------------------------------===//
__global__ 
void global_BottomUpBuild_ILP(ul current_offset, ul parent_offset,
                              ui number_of_node, node::Node* root);

/**
 *@brief creating branches
 */
std::vector<node::Branch> Tree::CreateBranches(std::shared_ptr<io::DataSet> input_data_set) {
  auto number_of_data = input_data_set->GetNumberOfData();
  auto points = input_data_set->GetPoints();

  // create branches
  std::vector<node::Branch> branches(number_of_data);

  for( int range(i, 0, number_of_data)) {
    branches[i].SetRect(&points[i*GetNumberOfDims()]);
  }

  return branches;
}

bool Tree::AssignHilbertIndexToBranches(std::vector<node::Branch> &branches) {
  unsigned int number_of_bits = (GetNumberOfDims()>2) ? 20:31;

  for(int range(i, 0, branches.size())) {
    auto points = branches[i].GetPoints();
    auto hilbert_index = mapper::Hilbert_Mapper::MappingIntoSingle(GetNumberOfDims(),
                                                                   number_of_bits, points);
    branches[i].SetIndex(hilbert_index);
  }

  return true;
}

std::vector<ui> Tree::GetLevelNodeCount(std::vector<node::Branch> &branches) {
  std::vector<ui> level_node_count;

  // in this case, previous level is real data not the leaf level
  ui current_level_nodes = branches.size();
  
  while(current_level_nodes > 1) {
    current_level_nodes = ((current_level_nodes%GetNumberOfDegrees())?1:0) 
                          + current_level_nodes/GetNumberOfDegrees();
    level_node_count.push_back(current_level_nodes);
  }
  return level_node_count;
}

ui Tree::GetTotalNodeCount(void) const{
  ui total_node_count=0;
  for( auto node_count  : level_node_count) {
    total_node_count+=node_count;
  }
  return total_node_count;
}

bool Tree::CopyToNode(std::vector<node::Branch> &branches, 
                      NodeType node_type,int level, ui offset) {
  ui branch_itr=0;

  while(branch_itr < branches.size()) {
    node_ptr[offset].SetBranch(branches[branch_itr++], branch_itr%GetNumberOfDegrees());

    // increase the node offset 
    if(!(branch_itr%GetNumberOfDegrees())){
      node_ptr[offset].SetNodeType(node_type);
      node_ptr[offset].SetLevel(level);
      offset++;
    }
  }

  node_ptr[offset].SetNodeType(node_type);
  node_ptr[offset].SetLevel(level);

  return true;
}

void Tree::SetChildPointers(node::Node* node_ptr, ui number_of_nodes) { 
  ui child_offset=1;
  for(ui range(node_itr, 0, number_of_nodes)) {
    auto branch_count = node_ptr[node_itr].GetBranchCount();
    for(ui range(branch_itr, 0, branch_count)) {
      node_ptr[node_itr].SetBranchChild(node_ptr+child_offset++, branch_itr);
   }
  }
}

void Tree::BottomUpBuild_ILP(ul current_offset, ul parent_offset, 
                             ui number_of_node, node::Node* root) {
  global_BottomUpBuild_ILP<<<GetNumberOfBlocks(), GetNumberOfThreads()>>>(current_offset, parent_offset, number_of_node, root);
}

__global__ 
void global_BottomUpBuild_ILP(ul current_offset, ul parent_offset, 
                              ui number_of_node, node::Node* root) {
  ui bid = blockIdx.x;
  ui tid = threadIdx.x;

  ui block_incremental_value = GetNumberOfBlocks();
  ui block_offset = bid;

  node::Node* current_node;
  node::Node* parent_node;

  while( block_offset < number_of_node ) {
    current_node = root+current_offset+block_offset;
    parent_node = root+parent_offset+(ul)(block_offset/GetNumberOfDegrees());

    parent_node->SetBranchChild(current_node, block_offset%GetNumberOfDegrees());
    parent_node->SetBranchIndex(current_node->GetLastBranchIndex(), block_offset%GetNumberOfDegrees());

    parent_node->SetLevel(current_node->GetLevel()-1);
    parent_node->SetBranchCount(GetNumberOfDegrees());
    parent_node->SetNodeType(NODE_TYPE_INTERNAL); 

    //Find out the min, max boundaries in this node and set up the parent rect.
    for( ui range(dim, 0, GetNumberOfDims())) {
      ui high_dim = dim+GetNumberOfDims();

      __shared__ float lower_boundary[GetNumberOfDegrees()];
      __shared__ float upper_boundary[GetNumberOfDegrees()];

      for( ui jump(thread, tid, GetNumberOfDegrees(), GetNumberOfThreads())) {
        if( thread < current_node->GetBranchCount()){
          lower_boundary[ thread ] = current_node->GetBranchPoint(thread,dim);
          upper_boundary[ thread ] = current_node->GetBranchPoint(thread,high_dim);
        } else {
          lower_boundary[ thread ] = 1.0f;
          upper_boundary[ thread ] = 0.0f;
        }
      }

      //threads in half get lower boundary

      int N = GetNumberOfDegrees()/2 + GetNumberOfDegrees()%2;
      while(N > 1){
        for( ui jump(thread, tid, N, GetNumberOfThreads())) {
          if(lower_boundary[thread] > lower_boundary[thread+N])
            lower_boundary[thread] = lower_boundary[thread+N];
        }
        N = N/2 + N%2;
        __syncthreads();
      }
      if(tid==0) {
        if(N==1) {
          if( lower_boundary[0] > lower_boundary[1])
            lower_boundary[0] = lower_boundary[1];
        }
      }
      //other half threads get upper boundary
      N = GetNumberOfDegrees()/2 + GetNumberOfDegrees()%2;
      while(N > 1){
        for( ui jump(thread, tid, N, GetNumberOfThreads())) {
          if(upper_boundary[thread] < upper_boundary[thread+N])
            upper_boundary[thread] = upper_boundary[thread+N];
        }
        N = N/2 + N%2;
        __syncthreads();
      }
      if(tid==0) {
        if(N==1) {
          if ( upper_boundary[0] < upper_boundary[1] )
            upper_boundary[0] = upper_boundary[1];
        }
      }

      if( tid == 0 ){
        parent_node->SetBranchPoint(lower_boundary[0], ( block_offset % GetNumberOfDegrees() ), dim);
        parent_node->SetBranchPoint(upper_boundary[0], ( block_offset % GetNumberOfDegrees() ), high_dim);
      }

      __syncthreads();
    }

    block_offset+=block_incremental_value;
  }

  //last node in each level
  if(  number_of_node % GetNumberOfDegrees() ){
    parent_node = root + current_offset - 1;
    if( number_of_node < GetNumberOfDegrees() ) {
      parent_node->SetBranchCount(number_of_node);
    }else{
      parent_node->SetBranchCount(number_of_node%GetNumberOfDegrees());
    }
  }

  // setting the node type for root at the end
  if( tid == 0 ) {
    root->SetNodeType(NODE_TYPE_ROOT);
  }
}

} // End of tree namespace
} // End of ursus namespace
