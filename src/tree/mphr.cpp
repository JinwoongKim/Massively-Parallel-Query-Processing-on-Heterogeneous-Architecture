#include "tree/mphr.h"

#include "common/macro.h"
#include "common/logger.h"
#include "evaluator/recorder.h"
#include "sort/thrust_sort.h"
#include "transformer/transformer.h"

#include <cassert>

namespace ursus {
namespace tree {

MPHR::MPHR() {
  tree_type = TREE_TYPE_MPHR;
}

/**
 * @brief build trees on the GPU
 * @param input_data_set 
 * @return true if success to build otherwise false
 */
bool MPHR::Build(std::shared_ptr<io::DataSet> input_data_set) {
  LOG_INFO("Build MPHR Tree");
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
 // Transform nodes into SOA fashion 
 //===--------------------------------------------------------------------===//
  node_soa_ptr = transformer::Transformer::Transform(node_ptr, total_node_count);
  assert(node_soa_ptr);

 //===--------------------------------------------------------------------===//
 // Move Trees to the GPU
 //===--------------------------------------------------------------------===//
  ret = MoveTreeToGPU(total_node_count);
  assert(ret);

  // FIXME :: REMOVE now it's only For debugging
  //PrintTree();
  //PrintTreeInSOA();

  // TODO Use smart pointer?
  free(node_ptr);
  free(node_soa_ptr);
  node_ptr = nullptr;
  node_soa_ptr = nullptr;

  return true;
}

int MPHR::Search(std::shared_ptr<io::DataSet> query_data_set, 
                   ui number_of_search) {
  auto& recorder = evaluator::Recorder::GetInstance();

  //===--------------------------------------------------------------------===//
  // Load Query 
  //===--------------------------------------------------------------------===//
  Point* d_query;
  cudaMalloc((void**) &d_query, sizeof(Point)*GetNumberOfDims()*2*number_of_search);
  auto query = query_data_set->GetPoints();
  cudaMemcpy(d_query, &query[0], sizeof(Point)*GetNumberOfDims()*2*number_of_search, cudaMemcpyHostToDevice);

  //===--------------------------------------------------------------------===//
  // Prepare Hit & Node Visit Variables for evaluations
  //===--------------------------------------------------------------------===//
  ui h_hit[GetNumberOfBlocks()] = {0};
  ui h_root_visit_count[GetNumberOfBlocks()] = {0};
  ui h_node_visit_count[GetNumberOfBlocks()] = {0};

  ui total_hit = 0;
  ui total_root_visit_count = 0;
  ui total_node_visit_count = 0;

  ui* d_hit;
  cudaMalloc((void**) &d_hit, sizeof(ui)*GetNumberOfBlocks());
  ui* d_root_visit_count;
  cudaMalloc((void**) &d_root_visit_count, sizeof(ui)*GetNumberOfBlocks());
  ui* d_node_visit_count;
  cudaMalloc((void**) &d_node_visit_count, sizeof(ui)*GetNumberOfBlocks());

  //===--------------------------------------------------------------------===//
  // Execute Search Function
  //===--------------------------------------------------------------------===//
  recorder.TimeRecordStart();

  ui number_of_batch = GetNumberOfBlocks();
  for(ui range(query_itr, 0, number_of_search, GetNumberOfBlocks())) {

    // if remaining query is less then number of blocks,
    // setting the number of cuda blocks as much as remaining query
    if(query_itr + GetNumberOfBlocks() > number_of_search) {
      number_of_batch = number_of_search - query_itr;
    }

    LOG_INFO("Execute MPRS with %u CUDA blocks", number_of_batch);
    global_RestartScanning_and_ParentCheck<<<number_of_batch,GetNumberOfThreads()>>>
           (&d_query[query_itr*GetNumberOfDims()*2], 
           d_hit, d_root_visit_count, d_node_visit_count);
    cudaMemcpy(h_hit, d_hit, sizeof(ui)*number_of_batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_root_visit_count, d_root_visit_count, sizeof(ui)*number_of_batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_node_visit_count, d_node_visit_count, sizeof(ui)*number_of_batch, cudaMemcpyDeviceToHost);

    for(ui range(i, 0, number_of_batch)) {
      total_hit += h_hit[i];
      total_root_visit_count += h_root_visit_count[i];
      total_node_visit_count += h_node_visit_count[i];
    }
  }
  cudaThreadSynchronize();
  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Search Time on the GPU = %.6fms", elapsed_time);

  //===--------------------------------------------------------------------===//
  // Show Results
  //===--------------------------------------------------------------------===//
  LOG_INFO("Hit : %u", total_hit);
  LOG_INFO("Root visit count : %u", total_root_visit_count);
  LOG_INFO("Node visit count : %u", total_node_visit_count);

  return true;
}

bool MPHR::Bottom_Up(std::vector<node::Branch> &branches) {
  auto& recorder = evaluator::Recorder::GetInstance();

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

 //===--------------------------------------------------------------------===//
 // Copy the leaf nodes to the GPU
 //===--------------------------------------------------------------------===//
  node_ptr = new node::Node[total_node_count];
  // Copy the branches to nodes 
  auto ret = CopyBranchToNode(branches, NODE_TYPE_LEAF, tree_height, leaf_node_offset);
  assert(ret);

  node::Node* d_node_ptr;
  cudaMalloc((void**) &d_node_ptr, sizeof(node::Node)*total_node_count);
  cudaMemcpy(d_node_ptr, node_ptr, sizeof(node::Node)*total_node_count, cudaMemcpyHostToDevice);
 //===--------------------------------------------------------------------===//
 // Construct the rest part of trees on the GPU
 //===--------------------------------------------------------------------===//
  recorder.TimeRecordStart();
  ul current_offset = total_node_count;
  for( ui range(level_itr, 0, tree_height)) {
    current_offset -= level_node_count[level_itr];
    ul parent_offset = (current_offset-level_node_count[level_itr+1]);
    BottomUpBuild_ILP(current_offset, parent_offset, level_node_count[level_itr], d_node_ptr);
  }
  // print out construction time on the GPU
  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Bottom-Up Construction Time on the GPU = %.6fs", elapsed_time/1000.0f);

  cudaMemcpy(node_ptr, d_node_ptr, sizeof(node::Node)*total_node_count, cudaMemcpyDeviceToHost);
  cudaFree(d_node_ptr);

  return true;
}

//===--------------------------------------------------------------------===//
// Cuda Function 
//===--------------------------------------------------------------------===//
/**
 * @brief execute MPRS algorithm 
 * @param 
 */
__global__ 
void global_RestartScanning_and_ParentCheck(Point* _query, ui* hit, 
                    ui* root_visit_count, ui* node_visit_count) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  __shared__ ui childOverlap[GetNumberOfDegrees()];
  __shared__ ui t_hit[GetNumberOfThreads()]; 
  __shared__ bool isHit;

  ui query_offset = bid*GetNumberOfDims()*2;
  __shared__ Point query[GetNumberOfDims()*2];

  if(tid < GetNumberOfDims()*2) {
    query[tid] = _query[query_offset+tid];
  }

  root_visit_count[bid] = 0;
  node_visit_count[bid] = 0;

  t_hit[tid] = 0;

  node::Node_SOA* root = g_node_soa_ptr;
  node::Node_SOA* node_soa_ptr = root;

  ul passed_hIndex = 0;
  ul last_hIndex = root->GetLastIndex();

  if( tid == 0 ) {
    root_visit_count[bid]++;
  }
  __syncthreads();

  while( passed_hIndex < last_hIndex ) {

    //look over the left most child node before reaching leaf node level
    while( node_soa_ptr->GetNodeType() == NODE_TYPE_INTERNAL ) {

      if( (tid < node_soa_ptr->GetBranchCount()) &&
          (node_soa_ptr->GetIndex(tid) > passed_hIndex) &&
          (node_soa_ptr->IsOverlap(query, tid))) {
        childOverlap[tid] = tid;
      } else {
        childOverlap[tid] = GetNumberOfDegrees()+1;
      }
      __syncthreads();


      // check if I am the leftmost
      // Gather the Overlap idex and compare
      int N = GetNumberOfDegrees()/2 + GetNumberOfDegrees()%2;
      while(N > 1){
        if ( tid < N ){
          if(childOverlap[tid] > childOverlap[tid+N] )  
            childOverlap[tid] = childOverlap[tid+N];
        }
        N = N/2+N%2;
        __syncthreads();
      }
      if( tid == 0) {
        if(N==1){
          if(childOverlap[0] > childOverlap[1])
            childOverlap[0] = childOverlap[1];
        }
      }
      __syncthreads();

      // none of the branches overlapped the query
      if( childOverlap[0] == ( GetNumberOfDegrees()+1)) {

        passed_hIndex = node_soa_ptr->GetLastIndex();
        node_soa_ptr = root;
        
        if(tid == 0){
          root_visit_count[bid]++;
        }
        break;
      }
      // there exists some overlapped node
      else{
        node_soa_ptr = node_soa_ptr+node_soa_ptr->GetChildOffset(childOverlap[0]);
        if( tid == 0 ) {
          node_visit_count[bid]++;
        }
     }
      __syncthreads();
    } // end of while loop for internal nodes


    while(node_soa_ptr->GetNodeType() == NODE_TYPE_LEAF) {
      isHit = false;

      if(tid < node_soa_ptr->GetBranchCount() &&
        node_soa_ptr->IsOverlap(query, tid)){

        t_hit[tid]++;
        isHit = true;
      }
      __syncthreads();

      passed_hIndex = node_soa_ptr->GetLastIndex();

      // current node is the last leaf node, terminate search function
      if(node_soa_ptr->GetLastIndex() == last_hIndex ) {
        break;
      } else if( isHit ) { // continue searching function by jumping next leaf node
        node_soa_ptr++;

        if( tid == 0 ) {
          node_visit_count[bid]++;
        }
        __syncthreads();
      } else { 
        // go back to the parent node to check wthether other child nodes are overlapped with given query
        // Since ChildOffset of leaf node is pointing its parent node,
        // we can use it to go back to the parent node
        node_soa_ptr = root+node_soa_ptr->GetChildOffset(0);

        if( tid == 0 ) {
          if( node_soa_ptr == root){
            root_visit_count[bid]++;
          }else{
            node_visit_count[bid]++; 
         }
        }
        __syncthreads();
      }
    } // end of leaf node checking
  }


  __syncthreads();
  int N = GetNumberOfDegrees()/2 + GetNumberOfDegrees()%2;

  while(N > 1) {
    if ( tid < N ) {
      t_hit[tid] = t_hit[tid] + t_hit[tid+N];
    }
    N = N/2 + N%2;
    __syncthreads();
  }

  if(tid==0) {
    if(N==1) {
      hit[bid] = t_hit[0] + t_hit[1];
    } else {
      hit[bid] = t_hit[0];
    }
  }
}

} // End of tree namespace
} // End of ursus namespace

