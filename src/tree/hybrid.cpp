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
  std::cout <<"index name : "<< index_name <<std::endl;
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
    // Build the internal nodes in a bottop-up fashion on the GPU
    //===--------------------------------------------------------------------===//
    ret = Bottom_Up(branches); 
    assert(ret);

    DumpToFile(index_name);
  }

  //===--------------------------------------------------------------------===//
  // Transform nodes into SOA fashion 
  //===--------------------------------------------------------------------===//
  // transform only leaf nodes
  auto leaf_node_offset = total_node_count-level_node_count[0];
  node_soa_ptr = transformer::Transformer::Transform(&node_ptr[leaf_node_offset], 
      level_node_count[0]);
  assert(node_soa_ptr);

 //===--------------------------------------------------------------------===//
 // Move Trees to the GPU
 //===--------------------------------------------------------------------===//
  // move only leaf nodes to the GPU
  ret = MoveTreeToGPU(0, level_node_count[0]);
  assert(ret);

  free(node_soa_ptr);
  node_soa_ptr = nullptr;

  return true;
}

bool Hybrid::DumpFromFile(std::string index_name) {

  FILE* index_file;
  index_file = fopen(index_name.c_str(),"rb");

  if(!index_file) {
    LOG_INFO("Failed to load an index file (%s)", index_name.c_str());
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
  node_ptr = new node::Node[total_node_count];

  // read an index 
  fread(node_ptr, sizeof(node::Node), total_node_count, index_file);
  fclose(index_file);

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Done, time = %.6fms", elapsed_time);

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
  // write an index
  fwrite(node_ptr, sizeof(node::Node), total_node_count, index_file);
  fclose(index_file);

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Done, time = %.6fms", elapsed_time);
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
  // Prepare Hit & Node Visit Variables for evaluations
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
  ui number_of_batch = 1;

  for(ui range(query_itr, 0, number_of_search)) {
    ull visited_leafIndex = 0;
    ui node_visit_count = 0;
    ui chunk_size = 512; // FIXME pass chunk size through command linux
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

ull Hybrid::TraverseInternalNodes(node::Node *node_ptr, Point* query, 
                                  ull visited_leafIndex, ui *node_visit_count) {
  ull start_node_offset=0;
  (*node_visit_count)++;

  // internal nodes
  if(node_ptr->GetNodeType() == NODE_TYPE_INTERNAL) {
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
  }
  // leaf nodes
  else {
    // FIXME it returns hilbert index but if we use large scale data, we need
    // to rethink about this one again
    return node_ptr->GetBranchIndex(0);
  }
  return start_node_offset;
}

//===--------------------------------------------------------------------===//
// Cuda Function 
//===--------------------------------------------------------------------===//

__global__ 
void global_ParallelScanning_Leafnodes(Point* _query, ull start_node_offset, 
                                       ui chunk_size, ui* hit, ui* node_visit_count) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  ui query_offset = bid*GetNumberOfDims()*2;
  __shared__ Point query[GetNumberOfDims()*2];
  __shared__ ui t_hit[GetNumberOfThreads()]; 

  if(tid < GetNumberOfDims()*2) {
    query[tid] = _query[query_offset+tid];
  }

  t_hit[tid] = 0;
  node_visit_count[bid] = 0;

  node::Node_SOA* first_leaf_node = g_node_soa_ptr;
  node::Node_SOA* node_soa_ptr = first_leaf_node + start_node_offset;
  node::Node_SOA* last_node_soa_ptr = node_soa_ptr + chunk_size - 1;

  ull visited_leafIndex = 0; // FIXME?
  ull last_leafIndex = last_node_soa_ptr->GetLastIndex();
  __syncthreads();

  while( visited_leafIndex < last_leafIndex ) {

    MasterThreadOnly {
      node_visit_count[bid]++;
    }

    if(tid < node_soa_ptr->GetBranchCount() &&
        node_soa_ptr->IsOverlap(query, tid)) {
      t_hit[tid]++;
    }
    __syncthreads();

    visited_leafIndex = node_soa_ptr->GetLastIndex();

    node_soa_ptr++;
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

