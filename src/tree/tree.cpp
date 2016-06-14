#include "tree/tree.h"

#include "common/macro.h"
#include "common/logger.h"
#include "evaluator/evaluator.h"
#include "evaluator/recorder.h"
#include "mapper/hilbert_mapper.h"
#include "mapper/kmeans_mapper.h"

#include <algorithm>
#include <cmath>
#include <cassert>
#include <functional>
#include <thread>
#include <utility>
#include <queue>

namespace ursus {
namespace tree {

//===--------------------------------------------------------------------===//
// Constructor/Destructor
//===--------------------------------------------------------------------===//

TreeType Tree::GetTreeType() const {
  return tree_type;
}

/**
 * @brief : return an index name base on input data set
 * @param : input_data_set
 * @return : index name
 */
std::string Tree::GetIndexName(std::shared_ptr<io::DataSet> input_data_set){

  auto data_type = input_data_set->GetDataType();
  auto cluster_type = input_data_set->GetClusterType();
  auto dataset_type = input_data_set->GetDataSetType();
  auto number_of_data = input_data_set->GetNumberOfData();
  auto dimensions = GetNumberOfDims();
  auto degrees = GetNumberOfDegrees();
  std::string number_of_data_str = std::to_string(number_of_data);

  if(number_of_data >= 1000000) {
    number_of_data /= 1000000;
    number_of_data_str=std::to_string(number_of_data)+"M";
  }

  std::string index_name =
  "./index_files/"+DataTypeToString(data_type)+"_"+DataSetTypeToString(dataset_type)+
  "_DATA_"+std::to_string(dimensions)+
  "_"+ClusterTypeToString(cluster_type)+"_" +
  "DIMS_"+number_of_data_str+"_"+
  TreeTypeToString(tree_type)+"_"+std::to_string(degrees)+"_DEGREES";

  return index_name;
}

bool Tree::Top_Down(std::vector<node::Branch> &branches) {
  std::vector<ui> level_node_count;
  auto& recorder = evaluator::Recorder::GetInstance();
  recorder.TimeRecordStart();

  node_ptr = CreateNode(branches, 0, branches.size()-1, 0, level_node_count);

  for(ui range( level_itr, 0, level_node_count.size() )) {
    LOG_INFO("Level[%u] %zd", level_itr, level_node_count[level_itr]);
  }

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Top-Down Construction Time on the CPU = %.6fs", elapsed_time/1000.0f);

  return true;
}

/**
 * @brief : find the split position between start/end offsets base on the
 *          common prefix so that we can reduce the overlap of MBBs
 * @ param : branches
 * @ param : start_offset
 * @ param : end_offset
 * @ return : split offset
 */
ui Tree::GetSplitOffset(std::vector<node::Branch> &branches,
                        ui start_offset, ui end_offset) {

  // Get the first and last hilbert indices
  ll first_code = branches[start_offset].GetIndex();
  ll last_code = branches[end_offset].GetIndex();

  if (first_code >= last_code)  {
    return 0;
  }

  // Calculate the number of highest bits that are the same
  // for all objects, using the count-leading-zeros intrinsic.
  ui common_prefix = __builtin_clzl(first_code ^ last_code);

  // Use binary search to find where the next bit differs.
  // Specifically, we are looking for the highest object that
  // shares more than commonPrefix bits with the start_offset one.
  ui split_offset = start_offset; // initial guess
  ui step = end_offset - start_offset;

  do {
    step = (step + 1) >> 1; // exponential decrease
    ui new_split_offset = split_offset + step; // proposed new position

    if (new_split_offset < end_offset)  {
      ll split_code = branches[new_split_offset].GetIndex();
      ui split_prefix = __builtin_clzl(first_code ^ split_code);

      if (split_prefix > common_prefix)
        split_offset = new_split_offset; // accept proposal
    }
  } while (step > 1);

  return split_offset;
}
 
std::vector<ui> Tree::GetSplitPosition(std::vector<node::Branch> &branches, 
                                       ui start_offset, ui end_offset) {
  std::vector<ui> split_position;
  std::queue<std::pair<ui,ui>> offset_queue;

  // initialize queue with start/end offsets
  offset_queue.emplace(std::make_pair(start_offset, end_offset));

  // insert start/end offsets to split position list
  split_position.emplace_back(start_offset);
  split_position.emplace_back(end_offset);

  // find split position as long as offset_queue isn't empty or 
  // need more split positions
  while( !offset_queue.empty() &&
         split_position.size() < GetNumberOfDegrees()) {

    // dequeue the offset
    auto offset = offset_queue.front();
    offset_queue.pop();

    // get the split offset 
    auto split_offset = GetSplitOffset(branches, offset.first/*start_offset*/, 
                                       offset.second/*end_offset*/);

    // skip the current split offset if zero 
    if( !split_offset )  {
      continue;
    }

    // store split position
    split_position.emplace_back(split_offset);

    // enqueue to split nodes
    // Do not split when it doesn't have child nodes enough
    if( split_offset-offset.first >= GetNumberOfDegrees()) {
      offset_queue.push(std::make_pair(offset.first, split_offset));
    }
    if( (offset.second-split_offset+1) >= GetNumberOfDegrees()) {
      offset_queue.push(std::make_pair(split_offset+1, offset.second));
    }
  }

  // sort the split positions before return it
  std::sort(split_position.begin(), split_position.end());
  return split_position;
}

node::Node* Tree::CreateNode(std::vector<node::Branch> &branches, 
                             ui start_offset, ui end_offset, int level, 
                             std::vector<ui>& level_node_count) {

  node::Node* node = new node::Node();
  // increase node counts
  total_node_count++;

  if( level_node_count.size() <= level ) {
    level_node_count.emplace_back(1);
  } else {
    level_node_count[level]++;
  }

  //===--------------------------------------------------------------------===//
  // Create a leaf node
  //===--------------------------------------------------------------------===//
  auto number_of_data = (end_offset-start_offset)+1;
  if( number_of_data <= GetNumberOfDegrees() )  {
    for(ui range(branch_itr, 0, number_of_data)) {
      auto offset = start_offset+branch_itr;
      node->SetBranch(branches[offset], branch_itr);
      node->SetBranchIndex(branch_itr, branches[offset].GetIndex());
      node->SetBranchChildOffset(branch_itr, 0);
    }
    node->SetBranchCount(number_of_data);
    node->SetNodeType(NODE_TYPE_LEAF);
    node->SetLevel(level);

  } else {
    //===--------------------------------------------------------------------===//
    // Create an internal node
    //===--------------------------------------------------------------------===//
    auto split_position = GetSplitPosition(branches, start_offset, end_offset);

    for(ui range(child_itr, 0, split_position.size()-1)) {
      auto child_node = CreateNode(branches, split_position[child_itr], split_position[child_itr+1], 
                                   level+1, level_node_count);
      split_position[child_itr+1] += 1;

      // calculate child node's MBB and set it 
      auto points = child_node->GetMBB();
      for(ui range(dim, 0, GetNumberOfDims()*2)) {
        node->SetBranchPoint(child_itr, points[dim], dim);
      }
      node->SetBranchIndex(child_itr, child_node->GetLastBranchIndex());

      ll child_offset = (ll)child_node-(ll)node;
      node->SetBranchChildOffset(child_itr, child_offset);
    }
    node->SetBranchCount(split_position.size()-1);
    node->SetNodeType(NODE_TYPE_INTERNAL);
    node->SetLevel(level);
  }

  return node;
}

bool Tree::Bottom_Up(std::vector<node::Branch> &branches) {
  auto& recorder = evaluator::Recorder::GetInstance();
  recorder.TimeRecordStart();

  std::string device_type = "GPU";

 //===--------------------------------------------------------------------===//
 // Configure trees
 //===--------------------------------------------------------------------===//
  // Get node count for each level
  auto level_node_count = GetLevelNodeCount(branches);
  auto tree_height = level_node_count.size();
  total_node_count = GetTotalNodeCount(level_node_count);
  // set the leaf node count
  auto leaf_node_count = level_node_count.back();
  auto leaf_node_offset = total_node_count - leaf_node_count;

  for(ui range( level_itr, 0, level_node_count.size() )) {
    LOG_INFO("Level %zd", level_node_count[level_itr]);
  }

  node_ptr = new node::Node[total_node_count];
  // Copy the branches to nodes 
  auto ret = CopyBranchToNode(branches, NODE_TYPE_LEAF, tree_height-1, leaf_node_offset);
  assert(ret);

  // Calculate index size and get used and total device memory space
  auto index_size = total_node_count*sizeof(node::Node);
  auto total = evaluator::Evaluator::GetTotalMem();
  auto used = evaluator::Evaluator::GetUsedMem();

  // if an index is larger than device memory
  if( (index_size+used)/(double)total > 1.0) {
    device_type = "CPU";
    const size_t number_of_threads = std::thread::hardware_concurrency();

    // parallel for loop using c++ std 11 
    {
      std::vector<std::thread> threads;
      ul current_offset = total_node_count;

      //Launch a group of threads
      for( ui level_itr=tree_height-1; level_itr >0; level_itr--) {
        current_offset -= level_node_count[level_itr];
        ul parent_offset = (current_offset-level_node_count[level_itr-1]);

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
    cudaErrCheck(cudaMalloc((void**) &d_node_ptr, sizeof(node::Node)*total_node_count));
    cudaErrCheck(cudaMemcpy(d_node_ptr, node_ptr, sizeof(node::Node)*total_node_count, cudaMemcpyHostToDevice));

    //===--------------------------------------------------------------------===//
    // Construct the rest part of trees on the GPU
    //===--------------------------------------------------------------------===//
    ul current_offset = total_node_count;
      for( ui level_itr=tree_height-1; level_itr>0; level_itr--) {
      current_offset -= level_node_count[level_itr];
      ul parent_offset = (current_offset-level_node_count[level_itr-1]);
      BottomUpBuild_ILP(current_offset, parent_offset, level_node_count[level_itr], d_node_ptr);
    }
    cudaErrCheck(cudaMemcpy(node_ptr, d_node_ptr, sizeof(node::Node)*total_node_count, cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaFree(d_node_ptr));
  }
  
  // print out bottom up construction time
  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Bottom-Up Construction Time on the %s = %.6fs", device_type.c_str(), elapsed_time/1000.0f);

  return true;
}

void Tree::PrintTree(ui offset, ui count) {
  LOG_INFO("Print Tree");

  std::queue<node::Node*> bfs_queue;
  ui node_itr = 0;
  ui print_count = 0;

  // push the root node
  bfs_queue.emplace(node_ptr);

  // if the queue is not empty,
  while(!bfs_queue.empty())  {
    // pop the first element and print it out
    auto& node = bfs_queue.front();
    bfs_queue.pop();

    if(node_itr++>=offset) {
      std::cout << *node << std::endl;
      print_count++;
    }

    // if it is an internal node, push it's child nodes
    if( node->GetNodeType() != NODE_TYPE_LEAF)  {
      for(ui range(child_itr, 0, node->GetBranchCount())) {
        auto child_node = node->GetBranchChildNode(child_itr);
        bfs_queue.emplace(child_node);
      }
    }

    // if count is not zero, then print node out only as much as count
    if( count && print_count == count ) break;
  }
}

void Tree::PrintTreeInSOA(ui offset, ui count) {
  LOG_INFO("Print Tree in SOA");

  ui node_soa_itr = offset;
  ui print_count=0;

  while(print_count < count) {
    LOG_INFO("node %p",&node_soa_ptr[node_soa_itr]);
    std::cout << node_soa_ptr[node_soa_itr++] << std::endl;
    print_count++;
  }
}

void Tree::Thread_SetRect(std::vector<node::Branch> &branches, std::vector<Point>& points, 
                                                         ui start_offset, ui end_offset) {
  for(ui range(offset, start_offset, end_offset)) {
    branches[offset].SetRect(&points[offset*GetNumberOfDims()]);
    branches[offset].SetIndex(offset+1);
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

  const size_t number_of_threads = std::thread::hardware_concurrency();


  // parallel for loop using c++ std 11 
  {
    std::vector<std::thread> threads;

    auto chunk_size = branches.size()/number_of_threads;
    auto start_offset = 0 ;
    auto end_offset = start_offset + chunk_size + branches.size()%number_of_threads;

    //Launch a group of threads
    for (ui range(thread_itr, 0, number_of_threads)) {
      threads.push_back(std::thread(&Tree::Thread_SetRect, this, 
                                    std::ref(branches), std::ref(points), start_offset, end_offset));

      start_offset = end_offset;
      end_offset += chunk_size;
    }

    //Join the threads with the main thread
    for(auto &thread : threads){
      thread.join();
    }
  }

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Create Branche Time on CPU (%zu threads) = %.6fs", number_of_threads, elapsed_time/1000.0f);

  return branches;
}

void Tree::Thread_Mapping(std::vector<node::Branch> &branches, ui start_offset, ui end_offset) {
  ui number_of_bits = (GetNumberOfDims()>2) ? 20:31;

  for(ui range(offset, start_offset, end_offset)) {
    auto points = branches[offset].GetPoints();
    auto hilbert_index = mapper::HilbertMapper::MappingIntoSingle(GetNumberOfDims(),
                                                                   number_of_bits, points);
    branches[offset].SetIndex(hilbert_index);
  }
}

bool Tree::AssignHilbertIndexToBranches(std::vector<node::Branch> &branches) {
  auto& recorder = evaluator::Recorder::GetInstance();
  recorder.TimeRecordStart();

  const size_t number_of_threads = std::thread::hardware_concurrency();

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
  LOG_INFO("Assign Hilbert Index Time on CPU (%zu threads)= %.6fs", number_of_threads, elapsed_time/1000.0f);
  return true;
}

bool Tree::ClusterBrancheUsingKmeans(std::vector<node::Branch> &branches) {
  auto& recorder = evaluator::Recorder::GetInstance();
  recorder.TimeRecordStart();

  auto ret = mapper::KmeansMapper::ClusteringBranches(branches, GetNumberOfDims());
  assert(ret);

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Clustering branches using K-means&Hilbert Curve = %.6fs", elapsed_time/1000.0f);
  return true;
}

// only works for bottom-up construction
std::vector<ui> Tree::GetLevelNodeCount(const std::vector<node::Branch> branches) {
  std::vector<ui> level_node_count;

  // in this case, previous level is real data not the leaf level
  ui current_level_nodes = branches.size();
  
  while(current_level_nodes > 1) {
    current_level_nodes = ((current_level_nodes%GetNumberOfDegrees())?1:0) 
                          + current_level_nodes/GetNumberOfDegrees();
    level_node_count.emplace(level_node_count.begin(), current_level_nodes);
  }
  return level_node_count;
}

ui Tree::GetTotalNodeCount(const std::vector<ui> level_node_count) const{
  ui total_node_count=0;
  for( auto node_count  : level_node_count) {
    total_node_count+=node_count;
  }
  return total_node_count;
}

ui Tree::GetNumberOfBlocks(void) const{
  return number_of_cuda_blocks;
}

void Tree::Thread_CopyBranchToNode(std::vector<node::Branch> &branches, 
                            NodeType node_type,int level, ui node_offset, 
                            ui start_offset, ui end_offset) {

  node_offset += start_offset/GetNumberOfDegrees();

  for(ui range(branch_itr, start_offset, end_offset)) {
    node_ptr[node_offset].SetBranch(branches[branch_itr], branch_itr%GetNumberOfDegrees());
    // increase the node offset 
    if(((branch_itr+1)%GetNumberOfDegrees())==0){
      node_ptr[node_offset].SetNodeType(node_type);
      node_ptr[node_offset].SetLevel(level);
      node_ptr[node_offset].SetBranchCount(GetNumberOfDegrees());
      node_offset++;
    }
  }
}

bool Tree::CopyBranchToNode(std::vector<node::Branch> &branches, 
                            NodeType node_type,int level, ui leaf_node_offset) {

  auto& recorder = evaluator::Recorder::GetInstance();
  recorder.TimeRecordStart();

  const size_t number_of_threads = std::thread::hardware_concurrency();

  // parallel for loop using c++ std 11 
  {
    std::vector<std::thread> threads;

    auto chunk_size = branches.size()/number_of_threads;
    auto start_offset = 0 ;
    auto end_offset = start_offset + chunk_size + branches.size()%number_of_threads;

    //Launch a group of threads
    for (ui range(thread_itr, 0, number_of_threads)) {
      threads.push_back(std::thread(&Tree::Thread_CopyBranchToNode, this, 
                        std::ref(branches), node_type, level, leaf_node_offset, 
                        start_offset, end_offset));

      start_offset = end_offset;
      end_offset += chunk_size;
    }

    //Join the threads with the main thread
    for(auto &thread : threads){
      thread.join();
    }

    ui node_offset = leaf_node_offset + branches.size()/GetNumberOfDegrees();
    node_ptr[node_offset].SetNodeType(node_type);
    node_ptr[node_offset].SetLevel(level);
    node_ptr[node_offset].SetBranchCount(branches.size()%GetNumberOfDegrees());
  }

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Copy Branch To Node Time on the CPU = %.6fs", elapsed_time/1000.0f);

  return true;
}

void Tree::Thread_CopyBranchToNodeSOA(std::vector<node::Branch> &branches, 
                            NodeType node_type,int level, ui node_offset, 
                            ui start_offset, ui end_offset) {
  node_offset += start_offset/GetNumberOfDegrees();

  for(ui range(branch_itr, start_offset, end_offset)) {
    auto points = branches[branch_itr].GetPoints();
    auto index = branches[branch_itr].GetIndex();
    auto child_offset = branches[branch_itr].GetChildOffset();

    // range from 0 to (degrees-1) 
    auto branch_offset = branch_itr%GetNumberOfDegrees();

    // set points in Node_SOA
    for(ui range(dim_itr, 0, GetNumberOfDims()*2)) {
      auto point_offset = dim_itr*GetNumberOfDegrees()+branch_offset;
      node_soa_ptr[node_offset].SetPoint(point_offset, points[dim_itr]);
    }

   // set the index and child offset
    node_soa_ptr[node_offset].SetIndex(branch_offset, index);
    node_soa_ptr[node_offset].SetChildOffset(branch_offset, child_offset);

    // set the node type and level
    if(!branch_offset) { 
      node_soa_ptr[node_offset].SetNodeType(node_type);
      node_soa_ptr[node_offset].SetLevel(level);
    }

    // increase the node offset 
    if((branch_offset+1)==GetNumberOfDegrees()) { 
      // also branch count
      node_soa_ptr[node_offset].SetBranchCount(GetNumberOfDegrees());
      node_offset++;
    }
  }
}

bool Tree::CopyBranchToNodeSOA(std::vector<node::Branch> &branches, 
                               NodeType node_type, int level, ui node_offset) {

  auto& recorder = evaluator::Recorder::GetInstance();
  recorder.TimeRecordStart();

  const size_t number_of_threads = std::thread::hardware_concurrency();

  // parallel for loop using c++ std 11 
  {
    std::vector<std::thread> threads;

    auto chunk_size = branches.size()/number_of_threads;
    auto start_offset = 0 ;
    auto end_offset = start_offset + chunk_size + branches.size()%number_of_threads;

    //Launch a group of threads
    for (ui range(thread_itr, 0, number_of_threads)) {
      threads.push_back(std::thread(&Tree::Thread_CopyBranchToNodeSOA, this, 
                        std::ref(branches), node_type, level, node_offset, 
                        start_offset, end_offset));

      start_offset = end_offset;
      end_offset += chunk_size;
    }

    //Join the threads with the main thread
    for(auto &thread : threads){
      thread.join();
    }
  }


  if(branches.size()%GetNumberOfDegrees()) {
    node_soa_ptr[node_offset+(branches.size()/GetNumberOfDegrees())].SetBranchCount(branches.size()%GetNumberOfDegrees());
  }

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Copy Branch To NodeSOA Time on the CPU = %.6fs", elapsed_time/1000.0f);

  return true;
}



void Tree::BottomUpBuild_ILP(ul current_offset, ul parent_offset, 
                             ui number_of_node, node::Node* root) {
  global_BottomUpBuild_ILP<<<GetNumberOfBlocks(), GetNumberOfThreads()>>>
                          (current_offset, parent_offset, number_of_node, 
                          root, number_of_cuda_blocks);
}



void Tree::BottomUpBuildonCPU(ul current_offset, ul parent_offset, 
                              ui number_of_node, node::Node* root, ui tid, ui number_of_threads) {

  node::Node* current_node;
  node::Node* parent_node;

  for(ui range(node_offset, tid, number_of_node, number_of_threads)) {
    current_node = root+current_offset+node_offset;
    parent_node = root+parent_offset+(ul)(node_offset/GetNumberOfDegrees());

    parent_node->SetBranchChildOffset(node_offset%GetNumberOfDegrees(), 
                                      (ll)current_node-(ll)parent_node);

    // store the parent node offset for MPHR-tree
    if( current_node->GetNodeType() == NODE_TYPE_LEAF) {
      current_node->SetBranchChildOffset(0, (ll)parent_node-(ll)current_node);
    }

    parent_node->SetBranchIndex(node_offset%GetNumberOfDegrees(), current_node->GetLastBranchIndex());

    parent_node->SetLevel(current_node->GetLevel()-1);
    parent_node->SetBranchCount(GetNumberOfDegrees());

    // Set the node type
    if(current_node->GetNodeType() == NODE_TYPE_LEAF) {
      parent_node->SetNodeType(NODE_TYPE_INTERNAL); 
    } else {
      parent_node->SetNodeType(NODE_TYPE_EXTENDLEAF); 
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

      parent_node->SetBranchPoint( (node_offset % GetNumberOfDegrees()), lower_boundary[0], dim);
      parent_node->SetBranchPoint( (node_offset % GetNumberOfDegrees()), upper_boundary[0], high_dim);
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

ui Tree::BruteForceSearchOnCPU(Point* query) {

  auto& recorder = evaluator::Recorder::GetInstance();
  const size_t number_of_cpu_threads = std::thread::hardware_concurrency();

  std::vector<ll> start_node_offset;
  ui hit=0;

  // parallel for loop using c++ std 11 
  {
    std::vector<std::thread> threads;
    std::vector<ll> thread_start_node_offset[number_of_cpu_threads];
    ui thread_hit[number_of_cpu_threads];

    auto chunk_size = total_node_count/number_of_cpu_threads;
    auto start_offset = 0 ;
    auto end_offset = start_offset + chunk_size + total_node_count%number_of_cpu_threads;

    //Launch a group of threads
    for (ui range(thread_itr, 0, number_of_cpu_threads)) {
      threads.push_back(std::thread(&Tree::Thread_BruteForce, this, 
                        query, std::ref(thread_start_node_offset[thread_itr]), 
                        std::ref(thread_hit[thread_itr]),
                        start_offset, end_offset));

      start_offset = end_offset;
      end_offset += chunk_size;
    }

    //Join the threads with the main thread
    for(auto &thread : threads){
      thread.join();
    }

    for(ui range(thread_itr, 0, number_of_cpu_threads)) {
      start_node_offset.insert( start_node_offset.end(), 
                                thread_start_node_offset[thread_itr].begin(), 
                                thread_start_node_offset[thread_itr].end()); 
      hit += thread_hit[thread_itr];
    }
  }

  //std::sort(start_node_offset.begin(), start_node_offset.end());
  //for( auto offset : start_node_offset) {
  //  LOG_INFO("start node offset %lu", offset);
  //}
  LOG_INFO("Hit on CPU : %u", hit);

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("BruteForce Scanning on the CPU (%u threads) = %.6fs", number_of_cpu_threads, elapsed_time/1000.0f);

  return hit;
}

void Tree::Thread_BruteForceInSOA(Point* query, std::vector<ll> &start_node_offset,
                             ui &hit, ui start_offset, ui end_offset) {
  hit = 0;
  for(ui range(node_itr, start_offset, end_offset)) {
    for(ui range(child_itr, 0, node_soa_ptr[node_itr].GetBranchCount())) {
      if( node_soa_ptr[node_itr].GetNodeType() == NODE_TYPE_LEAF) {
        if(node_soa_ptr[node_itr].IsOverlap(query, child_itr) ) {
          start_node_offset.emplace_back(node_itr);
          hit++;
        }else{
          LOG_INFO("node itr %u", node_itr);
          std::cout<<node_soa_ptr<<std::endl;
        }
      }
    }
  }
}

void Tree::Thread_BruteForce(Point* query, std::vector<ll> &start_node_offset,
                             ui &hit, ui start_offset, ui end_offset) {
  hit = 0;
  for(ui range(node_itr, start_offset, end_offset)) {
    for(ui range(child_itr, 0, node_ptr[node_itr].GetBranchCount())) {
      if( node_ptr[node_itr].GetNodeType() == NODE_TYPE_LEAF) {
        if(node_ptr[node_itr].IsOverlap(query, child_itr) ) {
          start_node_offset.emplace_back(node_itr);
          hit++;
        }else{
          LOG_INFO("node itr %u", node_itr);
          std::cout<<node_ptr<<std::endl;
        }
      }
    }
  }
}



//===--------------------------------------------------------------------===//
// Cuda Variable & Function 
//===--------------------------------------------------------------------===//
__global__ 
void global_BottomUpBuild_ILP(ul current_offset, ul parent_offset, 
                              ui number_of_node, node::Node* root,
                              ui number_of_cuda_blocks) {
  ui bid = blockIdx.x;
  ui tid = threadIdx.x;

  ui block_incremental_value = number_of_cuda_blocks;
  ui block_offset = bid;

  node::Node* current_node;
  node::Node* parent_node;

  while( block_offset < number_of_node ) {
    current_node = root+current_offset+block_offset;
    parent_node = root+parent_offset+(ul)(block_offset/GetNumberOfDegrees());

    parent_node->SetBranchChildOffset(block_offset%GetNumberOfDegrees(), 
                                     (ll)current_node-(ll)parent_node);

    MasterThreadOnly {
      // keep the parent node offset in order to go back to the parent node 
      if( current_node->GetNodeType() == NODE_TYPE_LEAF) {
        current_node->SetBranchChildOffset(0, (ll)parent_node-(ll)current_node);
        parent_node->SetNodeType(NODE_TYPE_EXTENDLEAF); 
      } else {
        parent_node->SetNodeType(NODE_TYPE_INTERNAL); 
      }

      parent_node->SetBranchIndex(block_offset%GetNumberOfDegrees(), current_node->GetLastBranchIndex());

      parent_node->SetLevel(current_node->GetLevel()-1);
      parent_node->SetBranchCount(GetNumberOfDegrees());
    }
    __syncthreads();

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
      // TODO :: Use macro parallel reduction

      int N = GetNumberOfDegrees()/2 + GetNumberOfDegrees()%2;
      while(N > 1){
        for( ui range(thread, tid, N, GetNumberOfThreads())) {
          if(lower_boundary[thread] > lower_boundary[thread+N])
            lower_boundary[thread] = lower_boundary[thread+N];
        }
        N = N/2 + N%2;
        __syncthreads();
      }
      MasterThreadOnly {
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
      MasterThreadOnly {
        if(N==1) {
          if ( upper_boundary[0] < upper_boundary[1] )
            upper_boundary[0] = upper_boundary[1];
        }
      }

      MasterThreadOnly{
        parent_node->SetBranchPoint( (block_offset % GetNumberOfDegrees()), lower_boundary[0], dim);
        parent_node->SetBranchPoint( (block_offset % GetNumberOfDegrees()), upper_boundary[0], high_dim);
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


} // End of tree namespace
} // End of ursus namespace
