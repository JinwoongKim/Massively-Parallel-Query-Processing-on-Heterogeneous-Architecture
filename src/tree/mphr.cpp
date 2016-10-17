#include "tree/mphr.h"

#include "common/macro.h"
#include "common/logger.h"
#include "evaluator/recorder.h"
#include "sort/sorter.h"
#include "transformer/transformer.h"
#include "manager/chunk_manager.h"

#include <cassert>

#include "cuda_profiler_api.h"

namespace ursus {
namespace tree {

MPHR::MPHR() {
  tree_type = TREE_TYPE_MPHR;
}

MPHR::~MPHR() {
  if( b_node_ptr != nullptr) {
    delete b_node_ptr;
  }
  if( node_soa_ptr != nullptr) {
    delete node_soa_ptr;
  }
}

/**
 * @brief build trees on the GPU
 * @param input_data_set 
 * @return true if success to build otherwise false
 */
bool MPHR::Build(std::shared_ptr<io::DataSet> input_data_set) {
  LOG_INFO("Build MPHR Tree");
  bool ret = false;

  // Load an index from file it exists
  // otherwise, build an index and dump it to file
  auto index_name = GetIndexName(input_data_set);
  if(input_data_set->IsRebuild() || !DumpFromFile(index_name))  {
    //===--------------------------------------------------------------------===//
    // Create branches
    //===--------------------------------------------------------------------===//
    std::vector<node::Branch> branches = CreateBranches(input_data_set);

    //===--------------------------------------------------------------------===//
    // Assign Hilbert Ids to branches
    //===--------------------------------------------------------------------===//
    ret = AssignHilbertIndexToBranches(branches);
    assert(ret);

    //===--------------------------------------------------------------------===//
    // Sort the branches either CPU or GPU depending on the size
    //===--------------------------------------------------------------------===//
    ret = sort::Sorter::Sort(branches);
    assert(ret);

    node::Node_SOA* node_soa_ptr_backup[number_of_partition];
    std::vector<ui> node_soa_ptr_size;
    ui tmp_device_node_count=0;
    auto chunk_size = branches.size()/number_of_partition;
    auto start_offset = 0 ;
    auto end_offset = start_offset + chunk_size + branches.size()%number_of_partition;

    for(ui range(partition_itr, 0, number_of_partition)) {
      std::vector<node::Branch> partitioned_branches;
      partitioned_branches.resize(end_offset-start_offset);

      // copy the branch start from start offset to end offset into temp branches
      // so that we can build an index without modification of existing build function
      std::move(branches.begin()+start_offset, branches.begin()+end_offset,
                partitioned_branches.begin());

      //===--------------------------------------------------------------------===//
      // Build the tree in a bottop-up fashion on the GPU
      //===--------------------------------------------------------------------===//
      // TODO We may pass TREE_TYPE so that we can set the child offset to some
      // useful data in leaf nodes 
      ret = Bottom_Up(partitioned_branches/*, tree_type*/);
      assert(ret);

      //===--------------------------------------------------------------------===//
      // Transform nodes into SOA fashion 
      //===--------------------------------------------------------------------===//
      node_soa_ptr = transformer::Transformer::Transform(b_node_ptr, device_node_count);
      assert(node_soa_ptr);

      // free the b_node_ptr
      delete b_node_ptr;
      b_node_ptr = nullptr;

      node_soa_ptr_backup[partition_itr] = node_soa_ptr;
      node_soa_ptr_size.emplace_back(device_node_count);
      root_offset[partition_itr] = tmp_device_node_count;
      tmp_device_node_count += device_node_count;

      start_offset = end_offset;
      end_offset += chunk_size;
    }
    device_node_count = tmp_device_node_count;

    node_soa_ptr = new node::Node_SOA[device_node_count];
    for(ui range(partition_itr, 0, number_of_partition)) {
      memcpy(&node_soa_ptr[root_offset[partition_itr]], node_soa_ptr_backup[partition_itr],
             sizeof(node::Node_SOA)*node_soa_ptr_size[partition_itr]);
    }

    DumpToFile(index_name);
  }

  //===--------------------------------------------------------------------===//
  // Set Root Offset per Each CUDA Block
  //===--------------------------------------------------------------------===//
  ll* d_root_offset;
  cudaErrCheck(cudaMalloc((void**) &d_root_offset, sizeof(ll)*GetNumberOfBlocks()));
  cudaErrCheck(cudaMemcpy(d_root_offset, root_offset, 
                          sizeof(ll)*GetNumberOfBlocks(), cudaMemcpyHostToDevice));
  global_SetRootOffset<<<1,GetNumberOfBlocks()>>>(d_root_offset);

  //===--------------------------------------------------------------------===//
  // Move Tree to the GPU in advance
  //===--------------------------------------------------------------------===//
  // copy the entire tree  to the GPU
  // Get Chunk Manager and initialize it
  auto& chunk_manager = manager::ChunkManager::GetInstance();
  chunk_manager.Init(sizeof(node::Node_SOA)*device_node_count);
  chunk_manager.CopyNode(node_soa_ptr, 0, device_node_count);

    // deallocate tree on the host
  delete node_soa_ptr;
  node_soa_ptr = nullptr;

  return true;
}

bool MPHR::DumpFromFile(std::string index_name){

  FILE* index_file = OpenIndexFile(index_name);
  if(index_file == nullptr) {
    return false;
  }

  auto& recorder = evaluator::Recorder::GetInstance();
  recorder.TimeRecordStart();

  // read number of partition
  fread(&number_of_partition, sizeof(ui), 1, index_file);

  // read root offset
  fread(&root_offset, sizeof(ll), number_of_partition, index_file);

  // read total node count
  fread(&device_node_count, sizeof(ui), 1, index_file);

  node_soa_ptr = new node::Node_SOA[device_node_count];
  // read nodes
  fread(node_soa_ptr, sizeof(node::Node_SOA), device_node_count, index_file);

  fclose(index_file);

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Done, time = %.6fs", elapsed_time/1000.0f);

  return true;
}

bool MPHR::DumpToFile(std::string index_name) {
  auto& recorder = evaluator::Recorder::GetInstance();
  recorder.TimeRecordStart();
  LOG_INFO("Dump an index into file (%s)...", index_name.c_str());

  // NOTE :: Use fwrite since it is fast
  FILE* index_file;
  index_file = fopen(index_name.c_str(),"wb");

  // write number of partition
  fwrite(&number_of_partition, sizeof(ui), 1, index_file);

  // write root offset
  fwrite(&root_offset, sizeof(ll), number_of_partition, index_file);

  // write total node count
  fwrite(&device_node_count, sizeof(ui), 1, index_file);

  // write nodes
  fwrite(node_soa_ptr, sizeof(node::Node_SOA), device_node_count, index_file);
  fclose(index_file);

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Done, time = %.6fs", elapsed_time/1000.0f);
  return true;
}

int MPHR::Search(std::shared_ptr<io::DataSet> query_data_set, 
                   ui number_of_search, ui number_of_repeat) {
cudaProfilerStart();
  auto& recorder = evaluator::Recorder::GetInstance();

  //===--------------------------------------------------------------------===//
  // Read Query 
  //===--------------------------------------------------------------------===//
  auto d_query = query_data_set->GetDeviceQuery(number_of_search);

  for(ui range(repeat_itr, 0, number_of_repeat)) {
    LOG_INFO("#%u) Evaluation", repeat_itr+1);
    //===--------------------------------------------------------------------===//
    // Prepare Hit & Node Visit Variables for evaluations
    //===--------------------------------------------------------------------===//
    ui h_hit[GetNumberOfBlocks()];
    ui h_root_visit_count[GetNumberOfBlocks()];
    ui h_node_visit_count[GetNumberOfBlocks()];

    ui total_hit = 0;
    ui total_root_visit_count = 0;
    ui total_node_visit_count = 0;

    ui* d_hit;
    cudaErrCheck(cudaMalloc((void**) &d_hit, sizeof(ui)*GetNumberOfBlocks()));
    ui* d_root_visit_count;
    cudaErrCheck(cudaMalloc((void**) &d_root_visit_count, sizeof(ui)*GetNumberOfBlocks()));
    ui* d_node_visit_count;
    cudaErrCheck(cudaMalloc((void**) &d_node_visit_count, sizeof(ui)*GetNumberOfBlocks()));

    //===--------------------------------------------------------------------===//
    // Execute Search Function
    //===--------------------------------------------------------------------===//
    recorder.TimeRecordStart();

    ui number_of_batch = GetNumberOfBlocks();
    for(ui range(query_itr, 0, number_of_search,0)) {
      // if remaining query is less then number of blocks,
      // setting the number of cuda blocks as much as remaining query
      if( number_of_partition > 1) {
        number_of_batch = GetNumberOfBlocks();
      } else if(query_itr + GetNumberOfBlocks() > number_of_search) {
        number_of_batch = number_of_search - query_itr;
      }

      global_RestartScanning_and_ParentCheck<<<number_of_batch,GetNumberOfThreads()>>>
        (&d_query[query_itr*GetNumberOfDims()*2], number_of_partition, d_hit, 
         d_root_visit_count, d_node_visit_count);
      cudaMemcpy(h_hit, d_hit, sizeof(ui)*number_of_batch, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_root_visit_count, d_root_visit_count, sizeof(ui)*number_of_batch, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_node_visit_count, d_node_visit_count, sizeof(ui)*number_of_batch, cudaMemcpyDeviceToHost);

      for(ui range(i, 0, number_of_batch)) {
        total_hit += h_hit[i];
        total_root_visit_count += h_root_visit_count[i];
        total_node_visit_count += h_node_visit_count[i];
      }
      if(number_of_partition == 1) {
        query_itr += GetNumberOfBlocks(); 
      } else {
        query_itr += 1;
      }
    }
    auto elapsed_time = recorder.TimeRecordEnd();
cudaProfilerStop();

    //===--------------------------------------------------------------------===//
    // Show Results
    //===--------------------------------------------------------------------===//
    LOG_INFO("Hit : %u", total_hit);
    LOG_INFO("Avg. Search Time on the GPU(ms) = \n%.6f", elapsed_time/(float)number_of_search);
    LOG_INFO("Avg. Root visit count : \n%f", total_root_visit_count/(float)number_of_search);
    LOG_INFO("Avg. Node visit count : \n%f\n", total_node_visit_count/(float)number_of_search);
  }

  return true;
}

void MPHR::SetNumberOfCUDABlocks(ui _number_of_cuda_blocks){
  number_of_cuda_blocks = _number_of_cuda_blocks;
  assert(number_of_cuda_blocks);
}

void MPHR::SetNumberOfPartition(ui _number_of_partition){
  number_of_partition = _number_of_partition;
  if( number_of_partition > 1) {
    tree_type = TREE_TYPE_MPHR_PARTITION;
  }
  assert(number_of_partition);
}

//===--------------------------------------------------------------------===//
// Cuda Function 
//===--------------------------------------------------------------------===//

__device__ ll g_root_offset[GetNumberOfMAXBlocks()];

__global__ 
void global_SetRootOffset(ll* root_offset) {
  int tid = threadIdx.x;
  g_root_offset[tid] = root_offset[tid];
}

/**
 * @brief execute MPRS algorithm 
 * @param 
 */
__global__ 
void global_RestartScanning_and_ParentCheck(Point* _query, ui number_of_partition, ui* hit, 
                    ui* root_visit_count, ui* node_visit_count) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  //__shared__ ui childOverlap[GetNumberOfLeafNodeDegrees()];
  ui childOverlap[GetNumberOfLeafNodeDegrees()];
  __shared__ ui t_hit[GetNumberOfThreads()]; 
  __shared__ bool isHit;

  ui query_offset=0;
  if(number_of_partition == 1 ) {
    query_offset = bid*GetNumberOfDims()*2;
  }
  __shared__ Point query[GetNumberOfDims()*2];

  if(tid < GetNumberOfDims()*2) {
    query[tid] = _query[query_offset+tid];
  }

  root_visit_count[bid] = 0;
  node_visit_count[bid] = 0;

  t_hit[tid] = 0;

  node::Node_SOA* root = manager::g_node_soa_ptr + g_root_offset[bid];
  node::Node_SOA* node_soa_ptr = root;

  ll visited_leafIndex = 0;
  ll last_leafIndex = root->GetLastIndex();


  MasterThreadOnly {
    root_visit_count[bid]++;
  }
  __syncthreads();

  while( visited_leafIndex < last_leafIndex ) {

    //look over the left most child node before reaching leaf node level
    while( node_soa_ptr->GetNodeType() != NODE_TYPE_LEAF ) { 
      if( (tid < node_soa_ptr->GetBranchCount()) &&
          (node_soa_ptr->GetIndex(tid) > visited_leafIndex) &&
          (node_soa_ptr->IsOverlap(query, tid))) {
        childOverlap[tid] = tid;
      } else {
        childOverlap[tid] = GetNumberOfLeafNodeDegrees()+1;
      }
      __syncthreads();


      // check if I am the leftmost
      // Gather the Overlap idex and compare
      FindMinOnGPU(childOverlap, GetNumberOfLeafNodeDegrees());

      // none of the branches overlapped the query
      if( childOverlap[0] == ( GetNumberOfLeafNodeDegrees()+1)) {

        visited_leafIndex = node_soa_ptr->GetLastIndex();
        node_soa_ptr = root;
        
        MasterThreadOnly {
          root_visit_count[bid]++;
        }
        break;
      }
      // there exists some overlapped node
      else{
        node_soa_ptr = node_soa_ptr->GetChildNode(childOverlap[0]);
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
        // Since the first child offset of a leaf node is pointing its parent node,
        // we can use it for back-tracking  
        node_soa_ptr = node_soa_ptr->GetChildNode(0);

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

  ParallelReduction(t_hit, GetNumberOfLeafNodeDegrees());

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

