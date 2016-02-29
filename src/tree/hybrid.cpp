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

 // TODO :: move branches into nodes
 //g_node_ptr = transformer::Transformer::Transform(node_ptr,node_ptr # of nodes);

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
  node_ptr = new node::Node[total_node_count];
  // Copy the branches to nodes 
  auto ret = CopyToNode(branches, NODE_TYPE_LEAF, tree_height, leaf_node_offset);
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

  // Re-set child pointers
  SetChildPointers(node_ptr, total_node_count-level_node_count[0]);
  return true;
}

void Hybrid::PrintTree(ui count) {
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

/*
__global__ 
void global_MPHR_ParentCheck(std::vector<Point> query, 
                             evaluator::Recorder recorder, 
                             std::vector<long> node_offsets){
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // Get a node offset based on the node type
  long leafNode_offset = node_offsets[NODE_TYPE_LEAF];
  long extendLeafNode_offset = node_offsets[NODE_TYPE_EXTENDLEAF];

  __shared__ int t_hit[GetNumberOfDegrees()]; 
  __shared__ int childOverlap[GetNumberOfDegrees()];
  __shared__ bool isHit;
  __shared__  Point query[GetNumberOfDims()]; // XXX Use rect plz

  G_Node* g_node_ptr;

  t_hit[tid] = 0;
  hit[bid] = 0;

  G_Node* root = (G_Node*) deviceRoot[partition_index];
  G_Node* leafNode_ptr = (G_Node*) ( (char*) root+(PGSIZE*leafNode_offset) );
  G_Node* extendNode_ptr = (G_Node*) ( (char*) root+(PGSIZE*extendLeafNode_offset) );
  __syncthreads();


  // TODO :: rename
  int passed_hIndex; 
  int last_hIndex;

  query = _query[bid];
  passed_hIndex = 0;
  last_hIndex   = root->index[root->count-1];

    g_node_ptr = root;
    if( tid == 0 )
    {
      rootCount[bid]++;
    }
    __syncthreads();

    while( passed_hIndex < last_hIndex )
    {//find out left most child node till leaf level
      while( g_node_ptr ->level > 0 ) {

        if( (tid < g_node_ptr ->count) &&
            (g_node_ptr ->index[tid]> passed_hIndex) &&
            (dev_Node_SOA_Overlap(&query, g_node_ptr , tid)))
        {
          childOverlap[tid] = tid;
        }
        else
        {
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
          passed_hIndex = g_node_ptr->index[g_node_ptr->count-1];

          g_node_ptr = root;
          if( tid == 0 )
          {
            rootCount[bid]++;
          }
          break;
        }
        // there exists some overlapped node
        else{
          g_node_ptr = g_node_ptr->child[ childOverlap[0] ];
          if( tid == 0 )
          {
              count[bid]++;
          }
        }
        __syncthreads();
      }


      while( g_node_ptr->level == 0 )
      {

        isHit = false;

        if ( tid < g_node_ptr->count && dev_Node_SOA_Overlap(&query, g_node_ptr, tid))
        {
          t_hit[tid]++;
          isHit = true;
        }
        __syncthreads();

        passed_hIndex = g_node_ptr->index[g_node_ptr->count-1];

        //last leaf node

        if ( g_node_ptr->index[g_node_ptr->count-1] == last_hIndex )
          break;
        else if( isHit )
        {
          g_node_ptr++;
          if( tid == 0 )
          {
            count[bid]++;
          }
          __syncthreads();
        }
        else
        {
          g_node_ptr = extendNode_ptr + ( ( g_node_ptr - leafNode_ptr) / GetNumberOfDegrees()) ;
          if( tid == 0 )
          {
            if( g_node_ptr == root)
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
  while(N > 1){
    if ( tid < N )
    {
      t_hit[tid] = t_hit[tid] + t_hit[tid+N];
    }

    N = N/2 + N%2;
    __syncthreads();
  }

  if(tid==0) {
    if(N==1) 
      hit[bid] = t_hit[0] + t_hit[1];
    else
      hit[bid] = t_hit[0];
  }

}

*/


} // End of tree namespace
} // End of ursus namespace

