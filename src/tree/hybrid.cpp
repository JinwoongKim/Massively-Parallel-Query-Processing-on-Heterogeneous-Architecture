#include "tree/hybrid.h"

#include "common/macro.h"
#include "common/logger.h"
#include "evaluator/recorder.h"
#include "sort/sorter.h"
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

  // Load an index from file it exists
  // otherwise, build an index and dump it to file
  auto index_name = GetIndexName(input_data_set);
  if(!DumpFromFile(index_name)) { 
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
    // Sort the branches either CPU or GPU depending on the size
    //===--------------------------------------------------------------------===//
    ret = sort::Sorter::Sort(branches);
    assert(ret);

    //===--------------------------------------------------------------------===//
    // Build the internal nodes in a top-down fashion on the GPU
    //===--------------------------------------------------------------------===//
    ret = Top_Down(branches); 
    assert(ret);

    //===--------------------------------------------------------------------===//
    // Transform nodes into SOA fashion 
    //===--------------------------------------------------------------------===//
    // transform only leaf nodes
    auto leaf_node_offset = total_node_count-level_node_count[0];
    node_soa_ptr = transformer::Transformer::Transform(&node_ptr[leaf_node_offset], 
        level_node_count[0]);
    assert(node_soa_ptr);

    // Dump internal and leaf nodes into a file
    DumpToFile(index_name);
  }

 //===--------------------------------------------------------------------===//
 // Move Trees to the GPU
 //===--------------------------------------------------------------------===//
  // move only leaf nodes to the GPU
  ret = MoveTreeToGPU(0, level_node_count[0]);
  assert(ret);

  //PrintTree(); 

  free(node_soa_ptr);
  node_soa_ptr = nullptr;

  return true;
}

bool Hybrid::DumpFromFile(std::string index_name) {

  FILE* index_file;
  index_file = fopen(index_name.c_str(),"rb");

  if(!index_file) {
    LOG_INFO("An index file(%s) doesn't exist", index_name.c_str());
    return false;
  }

  LOG_INFO("Load an index file (%s)", index_name.c_str());

  auto& recorder = evaluator::Recorder::GetInstance();

  size_t height;

  // read tree height
  fread(&height, sizeof(size_t), 1, index_file);
  level_node_count.resize(height);
  for(ui range(level_itr, 0, height)) {
    // read node count for each tree level
    fread(&level_node_count[level_itr], sizeof(ui), 1, index_file);
  }
  // read total node count
  fread(&total_node_count, sizeof(ui), 1, index_file);

  LOG_INFO("Number of nodes %u", total_node_count);

  node_ptr = new node::Node[total_node_count-level_node_count[0]];
  // read internal nodes
  fread(node_ptr, sizeof(node::Node), total_node_count-level_node_count[0], index_file);

  node_soa_ptr = new node::Node_SOA[level_node_count[0]];
  // read leaf nodes
  fread(node_soa_ptr, sizeof(node::Node_SOA), level_node_count[0], index_file);

  fclose(index_file);

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Done, time = %.6fs", elapsed_time/1000.0f);

  return true;
}

bool Hybrid::DumpToFile(std::string index_name) {
  auto& recorder = evaluator::Recorder::GetInstance();

  LOG_INFO("Dump an index into file (%s)...", index_name.c_str());

  // NOTE :: Use fwrite since it is fast
  FILE* index_file;
  index_file = fopen(index_name.c_str(),"wb");

  size_t height = level_node_count.size();
  // write tree height
  fwrite(&height, sizeof(size_t), 1, index_file);
  for(ui range(level_itr, 0, height)) {
    // write each tree node count
    fwrite(&level_node_count[level_itr], sizeof(ui), 1, index_file);
  }
  // write total node count
  fwrite(&total_node_count, sizeof(ui), 1, index_file);
  // write internal nodes
  fwrite(node_ptr, sizeof(node::Node), total_node_count-level_node_count[0], index_file);
  // write leaf nodes
  fwrite(node_soa_ptr, sizeof(node::Node_SOA), level_node_count[0], index_file);
  fclose(index_file);

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Done, time = %.6fs", elapsed_time/1000.0f);
  return true;
}

int Hybrid::Search(std::shared_ptr<io::DataSet> query_data_set, 
                   ui number_of_search){
  auto& recorder = evaluator::Recorder::GetInstance();

  //===--------------------------------------------------------------------===//
  // Read Query 
  //===--------------------------------------------------------------------===//
  auto query = query_data_set->GetPoints();
  auto d_query = query_data_set->GetDeviceQuery(number_of_search);

  //===--------------------------------------------------------------------===//
  // Prepare Hit & Node Visit Variables for an evaluation
  //===--------------------------------------------------------------------===//
  ui h_hit[GetNumberOfBlocks()] = {0};
  ui h_node_visit_count[GetNumberOfBlocks()] = {0};

  ui total_hit = 0;
  ui total_node_visit_count_cpu = 0;
  ui total_node_visit_count_gpu = 0;

  ui* d_hit;
  cudaMalloc((void**) &d_hit, sizeof(ui)*GetNumberOfBlocks());
  ui* d_node_visit_count;
  cudaMalloc((void**) &d_node_visit_count, sizeof(ui)*GetNumberOfBlocks());

  //===--------------------------------------------------------------------===//
  // Execute Search Function
  //===--------------------------------------------------------------------===//
  recorder.TimeRecordStart();
  // FIXME currently, we only use single CUDA block
  ui number_of_batch = GetNumberOfDegrees();

  for(ui range(query_itr, 0, number_of_search)) {
    ll visited_leafIndex = 0;
    ui node_visit_count = 0;
    ui query_offset = query_itr*GetNumberOfDims()*2;

    while(1) {
      //===--------------------------------------------------------------------===//
      // Traversal Internal Nodes on CPU
      //===--------------------------------------------------------------------===//
      auto start_node_hIndex = TraverseInternalNodes(node_ptr, &query[query_offset],
                                                     visited_leafIndex, &node_visit_count);

      auto start_node_offset = start_node_hIndex/GetNumberOfDegrees(); 
      total_node_visit_count_cpu += node_visit_count;

      // no more overlapping internal nodes, terminate current query
      if( start_node_hIndex == 0) {
        break;
      }

      // resize chunk_size if the sum of start node offset and chunk size is
      // larger than number of leaf nodes
      if(start_node_offset+chunk_size > level_node_count[0]) {
        chunk_size = level_node_count[0] - start_node_offset;
      }

      //===--------------------------------------------------------------------===//
      // Parallel Scanning Leaf Nodes on the GPU 
      //===--------------------------------------------------------------------===//
      global_ParallelScanning_Leafnodes<<<number_of_batch,GetNumberOfThreads()>>>
        (&d_query[query_offset], start_node_offset, chunk_size, d_hit, d_node_visit_count);

      cudaMemcpy(h_hit, d_hit, sizeof(ui)*number_of_batch, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_node_visit_count, d_node_visit_count, sizeof(ui)*number_of_batch, cudaMemcpyDeviceToHost);

      for(ui range(i, 0, number_of_batch)) {
        total_hit += h_hit[i];
        total_node_visit_count_gpu += h_node_visit_count[i];
      }
      visited_leafIndex = (start_node_offset+chunk_size)*GetNumberOfDegrees();
    }
  }
  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Search Time on the GPU = %.6fms", elapsed_time);

  //===--------------------------------------------------------------------===//
  // Show Results
  //===--------------------------------------------------------------------===//
  LOG_INFO("Hit : %u", total_hit);
  LOG_INFO("Node visit count on CPU : %u", total_node_visit_count_cpu);
  LOG_INFO("Node visit count on GPU : %u", total_node_visit_count_gpu);
}

void Hybrid::SetChunkSize(ui _chunk_size){
  chunk_size = _chunk_size;
}

ll Hybrid::TraverseInternalNodes(node::Node *node_ptr, Point* query, 
                                 ll visited_leafIndex, ui *node_visit_count) {
  ll start_node_offset=0;
  (*node_visit_count)++;

  // internal nodes
  if(node_ptr->GetNodeType() == NODE_TYPE_INTERNAL ) {
    for(ui range(branch_itr, 0, node_ptr->GetBranchCount())) {
      if( node_ptr->GetBranchIndex(branch_itr) > visited_leafIndex && 
          node_ptr->IsOverlap(query, branch_itr)) {
            start_node_offset=TraverseInternalNodes(node_ptr+node_ptr->GetBranchChildOffset(branch_itr), 
                                   query, visited_leafIndex, node_visit_count);
            if(start_node_offset > 0) {
              break;
            }
      }
    }
  } // extend leaf nodes
  else {
    // FIXME it returns hilbert index but if we use large scale data, we need
    // to rethink about this one again
    for(ui range(branch_itr, 0, node_ptr->GetBranchCount())) {
      if( node_ptr->GetBranchIndex(branch_itr) > visited_leafIndex && 
          node_ptr->IsOverlap(query, branch_itr)) {

        start_node_offset = node_ptr->GetBranchIndex(branch_itr);

        if( start_node_offset%GetNumberOfDegrees() != 0) {
          start_node_offset = start_node_offset%GetNumberOfDegrees();
        } else {
          start_node_offset = start_node_offset-GetNumberOfDegrees()+1;
        }

        return start_node_offset;
      }
    }
  }

  return start_node_offset;
}

//===--------------------------------------------------------------------===//
// Cuda Function 
//===--------------------------------------------------------------------===//

__global__ 
void global_ParallelScanning_Leafnodes(Point* _query, ll start_node_offset, 
                                       ui chunk_size, ui* hit, ui* node_visit_count) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  __shared__ Point query[GetNumberOfDims()*2];
  __shared__ ui t_hit[GetNumberOfThreads()]; 

  if(tid < GetNumberOfDims()*2) {
    query[tid] = _query[tid];
  }

  t_hit[tid] = 0;
  node_visit_count[bid] = 0;

  node::Node_SOA* first_leaf_node = g_node_soa_ptr;
  node::Node_SOA* node_soa_ptr = first_leaf_node + start_node_offset + bid;

  __syncthreads();

  for(ui range(node_itr, bid, chunk_size, GetNumberOfDegrees())) {

    MasterThreadOnly {
      node_visit_count[bid]++;
    }

    if(tid < node_soa_ptr->GetBranchCount() &&
        node_soa_ptr->IsOverlap(query, tid)) {
      t_hit[tid]++;
    }
    __syncthreads();

    node_soa_ptr+=GetNumberOfDegrees();
  }
  __syncthreads();

  //FIXME Do parallel reduction only last time
  //===--------------------------------------------------------------------===//
  // Parallel Reduction 
  //===--------------------------------------------------------------------===//
  ParallelReduction(t_hit, GetNumberOfThreads());

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

