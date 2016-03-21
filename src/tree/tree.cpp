#include "tree/tree.h"

#include "common/macro.h"
#include "common/logger.h"
#include "evaluator/evaluator.h"
#include "evaluator/recorder.h"
#include "mapper/hilbert_mapper.h"

#include <cmath>
#include <cassert>
#include <thread>
#include <functional>

namespace ursus {
namespace tree {

TreeType Tree::GetTreeType() const {
  return tree_type;
}

std::string Tree::GetIndexName(std::shared_ptr<io::DataSet> input_data_set){
  auto data_type = input_data_set->GetDataType();
  auto number_of_data = input_data_set->GetNumberOfData();
  auto dimensions = GetNumberOfDims();
  auto degrees = GetNumberOfDegrees();
  std::string number_of_data_str = std::to_string(number_of_data);

  if(number_of_data >= 1000000) {
    number_of_data /= 1000000;
    number_of_data_str=std::to_string(number_of_data)+"M";
  }

  std::string index_name = DataTypeToString(data_type)+"_DATA_"+std::to_string(dimensions)+"DIMS_"
 +number_of_data_str+"_"+TreeTypeToString(tree_type)+"_"+std::to_string(degrees)+"_DEGREES";

  return index_name;
}

bool Tree::Bottom_Up(std::vector<node::Branch> &branches) {
  auto& recorder = evaluator::Recorder::GetInstance();
  std::string device_type = "GPU";

 //===--------------------------------------------------------------------===//
 // Configure trees
 //===--------------------------------------------------------------------===//
  // Get node count for each level
  level_node_count = GetLevelNodeCount(branches);
  auto tree_height = level_node_count.size()-1;
  total_node_count = GetTotalNodeCount();
  auto leaf_node_offset = total_node_count - level_node_count[0];

  for(auto node_count : level_node_count) {
    LOG_INFO("Level : %u nodes", node_count);
  }

  node_ptr = new node::Node[total_node_count];
  // Copy the branches to nodes 
  auto ret = CopyBranchToNode(branches, NODE_TYPE_LEAF, tree_height, leaf_node_offset);
  assert(ret);

  recorder.TimeRecordStart();

  // Calculate index size and get used and total device memory space
  auto index_size = total_node_count*sizeof(node::Node);
  auto total = evaluator::Evaluator::GetTotalMem();
  auto used = evaluator::Evaluator::GetUsedMem();

  // if an index is larger than device memory
  if((index_size+used)/(double)total > 1.0) {
    device_type = "CPU";
    const size_t number_of_threads = std::thread::hardware_concurrency();

    // parallel for loop using c++ std 11 
    {
      std::vector<std::thread> threads;
      ul current_offset = total_node_count;

      //Launch a group of threads
      for( ui range(level_itr, 0, tree_height)) {
        current_offset -= level_node_count[level_itr];
        ul parent_offset = (current_offset-level_node_count[level_itr+1]);

        for (ui range(thread_itr, 0, number_of_threads)) {
          threads.push_back(std::thread(&Tree::BottomUpBuildonCPU, this, current_offset, 
                parent_offset, level_node_count[level_itr], std::ref(node_ptr), thread_itr, 
                number_of_threads));
        }
        //Join the threads with the main thread
        for(auto &thread : threads){
          thread.join();
        }
        threads.clear();
      }
   }
  } else {
    //===--------------------------------------------------------------------===//
    // Copy the leaf nodes to the GPU
    //===--------------------------------------------------------------------===//
    node::Node* d_node_ptr;
    cudaMalloc((void**) &d_node_ptr, sizeof(node::Node)*total_node_count);
    cudaMemcpy(d_node_ptr, node_ptr, sizeof(node::Node)*total_node_count, cudaMemcpyHostToDevice);

    //===--------------------------------------------------------------------===//
    // Construct the rest part of trees on the GPU
    //===--------------------------------------------------------------------===//
    ul current_offset = total_node_count;
    for( ui range(level_itr, 0, tree_height)) {
      current_offset -= level_node_count[level_itr];
      ul parent_offset = (current_offset-level_node_count[level_itr+1]);
      BottomUpBuild_ILP(current_offset, parent_offset, level_node_count[level_itr], d_node_ptr);
    }
    cudaMemcpy(node_ptr, d_node_ptr, sizeof(node::Node)*total_node_count, cudaMemcpyDeviceToHost);
    cudaFree(d_node_ptr);
  }

  // print out construction time on the GPU
  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Bottom-Up Construction Time on the %s = %.6fs", device_type.c_str(), elapsed_time/1000.0f);

  return true;
}

void Tree::PrintTree(ui offset, ui count) {
  LOG_INFO("Print Tree");
  LOG_INFO("Height %zu", level_node_count.size());

  ui node_itr=offset;

  for( int i=level_node_count.size()-1; i>=0; --i) {
    LOG_INFO("Level %zd", (level_node_count.size()-1)-i);
    for( ui range(j, 0, level_node_count[i])){
      LOG_INFO("node %p",&node_ptr[node_itr]);
      std::cout << node_ptr[node_itr++] << std::endl;

      if(count){ if( node_itr>=count){ return; } }
    }
  }
}

void Tree::PrintTreeInSOA(ui offset, ui count) {
  LOG_INFO("Print Tree in SOA");
  LOG_INFO("Height %zu", level_node_count.size());

  ui node_soa_itr=offset;

  for( int i=level_node_count.size()-1; i>=0; --i) {
    LOG_INFO("Level %zd", (level_node_count.size()-1)-i);
    for( ui range(j, 0, level_node_count[i])){
      LOG_INFO("node %p",&node_soa_ptr[node_soa_itr]);
      std::cout << node_soa_ptr[node_soa_itr++] << std::endl;

      if(count){ if( node_soa_itr>=count){ return; } }
    }
  }
}

/**
 *@brief creating branches
 */
std::vector<node::Branch> Tree::CreateBranches(std::shared_ptr<io::DataSet> input_data_set) {
  auto& recorder = evaluator::Recorder::GetInstance();
  recorder.TimeRecordStart();

  auto number_of_data = input_data_set->GetNumberOfData();
  auto points = input_data_set->GetPoints();

  // create branches
  std::vector<node::Branch> branches(number_of_data);

  for( int range(i, 0, number_of_data)) {
    branches[i].SetRect(&points[i*GetNumberOfDims()]);
  }

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Create Branche  Time = %.6fs", elapsed_time/1000.0f);

  return branches;
}

void Tree::Thread_Mapping(std::vector<node::Branch> &branches, ui start_offset, ui end_offset) {
  ui number_of_bits = (GetNumberOfDims()>2) ? 20:31;

  for(ui range(offset, start_offset, end_offset)) {
    auto points = branches[offset].GetPoints();
    auto hilbert_index = mapper::Hilbert_Mapper::MappingIntoSingle(GetNumberOfDims(),
                                                                   number_of_bits, points);
    branches[offset].SetIndex(hilbert_index);
  }
}

bool Tree::AssignHilbertIndexToBranches(std::vector<node::Branch> &branches) {
  auto& recorder = evaluator::Recorder::GetInstance();
  recorder.TimeRecordStart();

  const size_t number_of_threads = std::thread::hardware_concurrency();
  LOG_INFO("Create %zu threads for mapping hilbert index in parallel", number_of_threads);

  // parallel for loop using c++ std 11 
  {
    std::vector<std::thread> threads;

    auto chunk_size = branches.size()/number_of_threads;
    auto start_offset = 0 ;
    auto end_offset = start_offset + chunk_size + branches.size()%number_of_threads;

    //Launch a group of threads
    for (ui range(thread_itr, 0, number_of_threads)) {
      threads.push_back(std::thread(&Tree::Thread_Mapping, this, 
                                    std::ref(branches), start_offset, end_offset));

      start_offset = end_offset;
      end_offset += chunk_size;
    }

    //Join the threads with the main thread
    for(auto &thread : threads){
      thread.join();
    }
  }

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Assign Hilbert Index Time on CPU = %.6fs", elapsed_time/1000.0f);
  return true;
}

std::vector<ui> Tree::GetLevelNodeCount(const std::vector<node::Branch> &branches) {
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

bool Tree::CopyBranchToNode(std::vector<node::Branch> &branches, 
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

void Tree::BottomUpBuild_ILP(ul current_offset, ul parent_offset, 
                             ui number_of_node, node::Node* root) {
  global_BottomUpBuild_ILP<<<GetNumberOfBlocks(), GetNumberOfThreads()>>>
                          (current_offset, parent_offset, number_of_node, root);
}

/**
 * @brief move tree to the GPU
 * @return true if success otherwise false
 */ 
bool Tree::MoveTreeToGPU(ui offset, ui count){

 if( count == 0 ) {
   count = total_node_count;
 }

 node::Node_SOA* d_node_soa_ptr;
 cudaMalloc((void**) &d_node_soa_ptr, sizeof(node::Node_SOA)*count);
 cudaMemcpy(d_node_soa_ptr, &node_soa_ptr[offset], sizeof(node::Node_SOA)*count, 
            cudaMemcpyHostToDevice);

 global_MoveTreeToGPU<<<1,1>>>(d_node_soa_ptr, count);

 return true;
}

void Tree::BottomUpBuildonCPU(ul current_offset, ul parent_offset, 
                              ui number_of_node, node::Node* root, ui tid, ui number_of_threads) {

  node::Node* current_node;
  node::Node* parent_node;

  for(ui range(node_offset, tid, number_of_node, number_of_threads)) {
    current_node = root+current_offset+node_offset;
    parent_node = root+parent_offset+(ul)(node_offset/GetNumberOfDegrees());

    // parent_node->SetBranchChildOffset(node_offset%GetNumberOfDegrees(), current_offset+node_offset);
    // NOTE :: To get the offset from the root node, use the above line otherwise use the below line.
    // With the below line, you will get the child offset from the current node
    // TODO This is not the good code to see, it might be better to scrub the codes at some point.
    parent_node->SetBranchChildOffset(node_offset%GetNumberOfDegrees(), 
        (current_offset+node_offset)-(parent_offset+(ul)(node_offset/GetNumberOfDegrees())));

    // store the parent node offset for MPHR-tree
    if( current_node->GetNodeType() == NODE_TYPE_LEAF) {
      current_node->SetBranchChildOffset(0, parent_offset+(ul)(node_offset/GetNumberOfDegrees()));
    }

    parent_node->SetBranchIndex(current_node->GetLastBranchIndex(), node_offset%GetNumberOfDegrees());

    parent_node->SetLevel(current_node->GetLevel()-1);
    parent_node->SetBranchCount(GetNumberOfDegrees());

    // Set the node type
    if(current_node->GetNodeType() == NODE_TYPE_LEAF) {
      parent_node->SetNodeType(NODE_TYPE_EXTENDLEAF); 
    } else {
      parent_node->SetNodeType(NODE_TYPE_INTERNAL); 
    }

    //Find out the min, max boundaries in this node and set up the parent rect.
    for(ui range(dim, 0, GetNumberOfDims())) {
      ui high_dim = dim+GetNumberOfDims();

      float lower_boundary[GetNumberOfDegrees()];
      float upper_boundary[GetNumberOfDegrees()];

      for( ui range(thread, 0, GetNumberOfDegrees())) {
        if( thread < current_node->GetBranchCount()){
          lower_boundary[ thread ] = current_node->GetBranchPoint(thread, dim);
          upper_boundary[ thread ] = current_node->GetBranchPoint(thread, high_dim);
        } else {
          lower_boundary[ thread ] = 1.0f;
          upper_boundary[ thread ] = 0.0f;
        }
      }

      //threads in half get lower boundary

      int N = GetNumberOfDegrees()/2 + GetNumberOfDegrees()%2;
      while(N > 1){
        for( ui range(thread, 0, N)) {
          if(lower_boundary[thread] > lower_boundary[thread+N])
            lower_boundary[thread] = lower_boundary[thread+N];
        }
        N = N/2 + N%2;
      }
      if(N==1) {
        if( lower_boundary[0] > lower_boundary[1])
          lower_boundary[0] = lower_boundary[1];
      }
      //other half threads get upper boundary
      N = GetNumberOfDegrees()/2 + GetNumberOfDegrees()%2;
      while(N > 1){
        for( ui range(thread, 0, N )) {
          if(upper_boundary[thread] < upper_boundary[thread+N])
            upper_boundary[thread] = upper_boundary[thread+N];
        }
        N = N/2 + N%2;
      }
      if(N==1) {
        if ( upper_boundary[0] < upper_boundary[1] )
          upper_boundary[0] = upper_boundary[1];
      }

      parent_node->SetBranchPoint(lower_boundary[0], ( node_offset % GetNumberOfDegrees() ), dim);
      parent_node->SetBranchPoint(upper_boundary[0], ( node_offset % GetNumberOfDegrees() ), high_dim);
    }
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
}


//===--------------------------------------------------------------------===//
// Cuda Variable & Function 
//===--------------------------------------------------------------------===//
__device__ node::Node_SOA* g_node_soa_ptr;

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

    // parent_node->SetBranchChildOffset(block_offset%GetNumberOfDegrees(), current_offset+block_offset);
    // NOTE :: To get the offset from the root node, use the above line otherwise use the below line.
    // With the below line, you will get the child offset from the current node
    // TODO This is not the good code to see, it might be better to scrub the codes at some point.
    parent_node->SetBranchChildOffset(block_offset%GetNumberOfDegrees(), 
                                     (current_offset+block_offset)-(parent_offset+(ul)(block_offset/GetNumberOfDegrees())));

    // keep the parent node offset in order to go back to the parent node 
    if( current_node->GetNodeType() == NODE_TYPE_LEAF) {
      current_node->SetBranchChildOffset(0, parent_offset+(ul)(block_offset/GetNumberOfDegrees()));
    }

    parent_node->SetBranchIndex(current_node->GetLastBranchIndex(), block_offset%GetNumberOfDegrees());

    parent_node->SetLevel(current_node->GetLevel()-1);
    parent_node->SetBranchCount(GetNumberOfDegrees());

    // Set the node type
    if(current_node->GetNodeType() == NODE_TYPE_LEAF) {
      parent_node->SetNodeType(NODE_TYPE_EXTENDLEAF); 
    } else {
      parent_node->SetNodeType(NODE_TYPE_INTERNAL); 
    }



    //Find out the min, max boundaries in this node and set up the parent rect.
    for( ui range(dim, 0, GetNumberOfDims())) {
      ui high_dim = dim+GetNumberOfDims();

      __shared__ float lower_boundary[GetNumberOfDegrees()];
      __shared__ float upper_boundary[GetNumberOfDegrees()];

      for( ui range(thread, tid, GetNumberOfDegrees(), GetNumberOfThreads())) {
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
        for( ui range(thread, tid, N, GetNumberOfThreads())) {
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
        for( ui range(thread, tid, N, GetNumberOfThreads())) {
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
}

__global__ 
void global_MoveTreeToGPU(node::Node_SOA* d_node_soa_ptr, ui total_node_count) { 
  g_node_soa_ptr = (node::Node_SOA*)malloc (sizeof(node::Node_SOA)*total_node_count);
  g_node_soa_ptr = d_node_soa_ptr;
}

} // End of tree namespace
} // End of ursus namespace
