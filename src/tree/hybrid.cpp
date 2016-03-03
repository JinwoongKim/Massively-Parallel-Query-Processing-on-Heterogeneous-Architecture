#include "tree/hybrid.h"

#include "common/macro.h"
#include "common/logger.h"
#include "evaluator/recorder.h"
#include "sort/thrust_sort.h"
#include "transformer/transformer.h"

#include <cassert>

namespace ursus {
namespace tree {

Hybrid::Hybrid() {
  tree_type = TREE_TYPE_HYBRID;
}

/**
 * @brief build trees on the GPU
 * @param input_data_set 
 * @return true if success to build otherwise false
 */
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
 // Transform nodes into SOA fashion 
 //===--------------------------------------------------------------------===//
  node_soa_ptr = transformer::Transformer::Transform(node_ptr,total_node_count);
  assert(node_soa_ptr);

 //===--------------------------------------------------------------------===//
 // Move Trees to the GPU
 //===--------------------------------------------------------------------===//
  ret = MoveTreeToGPU();
  assert(ret);

  // TODO :: REMOVE now it's only For debugging
  //PrintTree();
  //PrintTreeInSOA();

  return true;
}

int Hybrid::Search(std::shared_ptr<io::DataSet> query_data_set, 
                   ui number_of_search){
  
  std::vector<Point> query;
  evaluator::Recorder* d_recorder;
  std::vector<long> node_offsets;

  cudaMalloc((void**) &d_recorder, sizeof(evaluator::Recorder));

  //global_RestartScanning_and_ParentCheck<<<1,1>>>(query, d_recorder, node_offsets);

  return -1  ;
}

bool Hybrid::Bottom_Up(std::vector<node::Branch> &branches) {
  auto& recorder = evaluator::Recorder::GetInstance();

 //===--------------------------------------------------------------------===//
 // Configure trees
 //===--------------------------------------------------------------------===//
  // Get node count for each level
  level_node_count = GetLevelNodeCount(branches);
  auto tree_height = level_node_count.size()-1;
  total_node_count = GetTotalNodeCount();
  auto leaf_node_offset = total_node_count-level_node_count[0];;

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
  // print out sorting time on the GPU
  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("BottomUp Construction Time on the GPU = %.6fs", elapsed_time/1000.0f);

  cudaMemcpy(node_ptr, d_node_ptr, sizeof(node::Node)*total_node_count, cudaMemcpyDeviceToHost);
  cudaFree(d_node_ptr);

  return true;
}

void Hybrid::PrintTree(ui count) {
  LOG_INFO("Print Tree");
  LOG_INFO("Height %zu", level_node_count.size());

  ui node_itr=0;

  for( int i=level_node_count.size()-1; i>=0; --i) {
    LOG_INFO("Level %zd", (level_node_count.size()-1)-i);
    for( ui range(j, 0, level_node_count[i])){
      LOG_INFO("node %p",&node_ptr[node_itr]);
      std::cout << node_ptr[node_itr++] << std::endl;

      if(count){ if( node_itr>=count){ return; } }
    }
  }
}

void Hybrid::PrintTreeInSOA(ui count) {
  LOG_INFO("Print Tree in SOA");
  LOG_INFO("Height %zu", level_node_count.size());

  ui node_soa_itr=0;

  for( int i=level_node_count.size()-1; i>=0; --i) {
    LOG_INFO("Level %zd", (level_node_count.size()-1)-i);
    for( ui range(j, 0, level_node_count[i])){
      LOG_INFO("node %p",&node_soa_ptr[node_soa_itr]);
      std::cout << node_soa_ptr[node_soa_itr++] << std::endl;

      if(count){ if( node_soa_itr>=count){ return; } }
    }
  }
}

//===--------------------------------------------------------------------===//
// Cuda Function 
//===--------------------------------------------------------------------===//
/**
 * @brief execute MPRS algorithm 
 * @param recorder recording 
 */
 /*
__global__ 
void global_RestartScanning_and_ParentCheck(std::vector<Point> query, 
                                            evaluator::Recorder* recorder, 
                                            std::vector<long> node_offsets) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // Get node offsets based on the node type
  ul leafNode_offset = node_offsets[0];
  ul extendLeafNode_offset = node_offsets[1];


  __shared__ ui t_hit[GetNumberOfDegrees()]; 
  __shared__ ui childOverlap[GetNumberOfDegrees()];
  __shared__ Point query[GetNumberOfDims()]; // XXX Use rect plz
  __shared__ bool isHit;

  node::Node_SOA* node_soa_ptr;

  t_hit[tid] = 0;
  hit[bid] = 0;

  Node_SOA* root = (Node_SOA*) deviceRoot[partition_index];
  Node_SOA* leafNode_ptr = (Node_SOA*) ( (char*) root+(PGSIZE*leafNode_offset) );
  Node_SOA* extendNode_ptr = (Node_SOA*) ( (char*) root+(PGSIZE*extendLeafNode_offset) );
  __syncthreads();

  // TODO :: rename
  int passed_hIndex; 
  int last_hIndex;

  query = _query[bid];
  passed_hIndex = 0;
  last_hIndex   = root->index[root->count-1];

    node_soa_ptr = root;
    if( tid == 0 ) {
      rootCount[bid]++;
    }
    __syncthreads();

    while( passed_hIndex < last_hIndex ) {

      //find out left most child node till leaf level
      while( node_soa_ptr ->level > 0 ) {

        if( (tid < node_soa_ptr ->count) &&
            (node_soa_ptr ->index[tid]> passed_hIndex) &&
            (dev_Node_SOA_Overlap(&query, node_soa_ptr , tid))) {
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
          passed_hIndex = node_soa_ptr->index[node_soa_ptr->count-1];

          node_soa_ptr = root;

          if( tid == 0 ) {
            rootCount[bid]++;
          }
          break;
        }
        // there exists some overlapped node
        else{
          node_soa_ptr = node_soa_ptr->child[ childOverlap[0] ];

          if( tid == 0 ) {
              count[bid]++;
          }
        }
        __syncthreads();
      }


      while( node_soa_ptr->level == 0 ) {

        isHit = false;

        if ( tid < node_soa_ptr->count && dev_Node_SOA_Overlap(&query, node_soa_ptr, tid)) {
          t_hit[tid]++;
          isHit = true;
        }
        __syncthreads();

        passed_hIndex = node_soa_ptr->index[node_soa_ptr->count-1];

        //last leaf node

        if ( node_soa_ptr->index[node_soa_ptr->count-1] == last_hIndex )
          break;
        else if( isHit )
        {
          node_soa_ptr++;
          if( tid == 0 )
          {
            count[bid]++;
          }
          __syncthreads();
        }
        else
        {
          node_soa_ptr = extendNode_ptr + ( ( node_soa_ptr - leafNode_ptr) / GetNumberOfDegrees()) ;
          if( tid == 0 )
          {
            if( node_soa_ptr == root)
              rootCount[bid]++;
            else
              count[bid]++;
          }
          __syncthreads();
        }
      }
    }
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

*/

} // End of tree namespace
} // End of ursus namespace

