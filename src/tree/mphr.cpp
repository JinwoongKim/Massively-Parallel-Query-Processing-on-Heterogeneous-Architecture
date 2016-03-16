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
  // TODO We may pass TREE_TYPE so that we can set the child offset to some useful data in leaf nodes 
  ret = Bottom_Up(branches/*, tree_type*/);
  assert(ret);

 //===--------------------------------------------------------------------===//
 // Transform nodes into SOA fashion 
 //===--------------------------------------------------------------------===//
  node_soa_ptr = transformer::Transformer::Transform(node_ptr, total_node_count);
  assert(node_soa_ptr);

 //===--------------------------------------------------------------------===//
 // Move Trees to the GPU
 //===--------------------------------------------------------------------===//
  // copy the entire tree from the root node
  ret = MoveTreeToGPU();
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
  // Read Query 
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

  ull visited_leafIndex = 0;
  ull last_leafIndex = root->GetLastIndex();

  MasterThreadOnly {
    root_visit_count[bid]++;
  }
  __syncthreads();

  while( visited_leafIndex < last_leafIndex ) {

    //look over the left most child node before reaching leaf node level
    while( node_soa_ptr->GetNodeType() == NODE_TYPE_INTERNAL ) {

      if( (tid < node_soa_ptr->GetBranchCount()) &&
          (node_soa_ptr->GetIndex(tid) > visited_leafIndex) &&
          (node_soa_ptr->IsOverlap(query, tid))) {
        childOverlap[tid] = tid;
      } else {
        childOverlap[tid] = GetNumberOfDegrees()+1;
      }
      __syncthreads();


      // check if I am the leftmost
      // Gather the Overlap idex and compare
      FindLeftMostOverlappingChild(childOverlap, GetNumberOfDegrees());

      // none of the branches overlapped the query
      if( childOverlap[0] == ( GetNumberOfDegrees()+1)) {

        visited_leafIndex = node_soa_ptr->GetLastIndex();
        node_soa_ptr = root;
        
        MasterThreadOnly {
          root_visit_count[bid]++;
        }
        break;
      }
      // there exists some overlapped node
      else{
        node_soa_ptr = node_soa_ptr+node_soa_ptr->GetChildOffset(childOverlap[0]);
        MasterThreadOnly {
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

      visited_leafIndex = node_soa_ptr->GetLastIndex();

      // current node is the last leaf node, terminate search function
      if(node_soa_ptr->GetLastIndex() == last_leafIndex ) {
        break;
      } else if( isHit ) { // continue searching function by jumping next leaf node
        node_soa_ptr++;

        MasterThreadOnly {
          node_visit_count[bid]++;
        }
        __syncthreads();
      } else { 
        // go back to the parent node to check wthether other child nodes are overlapped with given query
        // Since ChildOffset of leaf node is pointing its parent node,
        // we can use it to go back to the parent node
        node_soa_ptr = root+node_soa_ptr->GetChildOffset(0);

        MasterThreadOnly {
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

  ParallelReduction(t_hit, GetNumberOfDegrees());

  MasterThreadOnly {
    if(N==1) {
      hit[bid] = t_hit[0] + t_hit[1];
    } else {
      hit[bid] = t_hit[0];
    }
  }
}

} // End of tree namespace
} // End of ursus namespace

