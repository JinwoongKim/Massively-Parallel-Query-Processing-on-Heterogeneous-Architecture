#include "tree/hybrid.h"

#include "common/macro.h"
#include "common/logger.h"
#include "evaluator/recorder.h"
#include "sort/sorter.h"
#include "transformer/transformer.h"
#include "manager/chunk_manager.h"

#include <cassert>
#include <thread>
#include <algorithm>

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
bool Hybrid::Build(std::shared_ptr<io::DataSet> input_data_set) {

  SetNumberOfNodeSOA(input_data_set->GetNumberOfData());

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
    ret = AssignHilbertIndexToBranches(branches);
    assert(ret);

    //===--------------------------------------------------------------------===//
    // Sort the branches either CPU or GPU depending on the size
    //===--------------------------------------------------------------------===//
    ret = sort::Sorter::Sort(branches);
    assert(ret);

    //===--------------------------------------------------------------------===//
    // Build the internal nodes in a top-down fashion 
    //===--------------------------------------------------------------------===//
    ret = Top_Down(branches); 
    assert(ret);

    //===--------------------------------------------------------------------===//
    // Build the last two level nodes in a bottom-up fashion 
    //===--------------------------------------------------------------------===//
    node_soa_ptr = new node::Node_SOA[GetNumberOfNodeSOA()];
    assert(node_soa_ptr);

    // Copy leaf nodes 
    ret = CopyBranchToNodeSOA(branches, NODE_TYPE_LEAF, 
                              1, GetNumberOfExtendLeafNodeSOA()/*offset*/);
    assert(ret);

    ret = BuildExtendLeafNodeOnCPU();
    assert(ret);

    // Dump an index to the file
    DumpToFile(index_name);
  }


  //===--------------------------------------------------------------------===//
  // Move Trees to the GPU in advance
  //===--------------------------------------------------------------------===//
  auto& chunk_manager = manager::ChunkManager::GetInstance();

  ui offset = 0;
  ui count = 0;
  
  if(scan_type == SCAN_TYPE_LEAF){
    // Move leaf nodes on the GPU
    offset = GetNumberOfExtendLeafNodeSOA();
    count = GetNumberOfLeafNodeSOA();
  }else if(scan_type == SCAN_TYPE_EXTENDLEAF) {
    // Move extend and leaf nodes on the GPU
    offset = 0;
    count = GetNumberOfNodeSOA();
  }else {
    LOG_INFO("scan type %s", (ScanTypeToString(scan_type)).c_str());
    assert(0);
  }

  // Get Chunk Manager and initialize it
  chunk_manager.Init(sizeof(node::Node_SOA)*count);
  chunk_manager.CopyNode(node_soa_ptr+offset, 0, count);

  LOG_INFO("Extend Leaf Node Count %u", GetNumberOfExtendLeafNodeSOA());
  LOG_INFO("Leaf Node Count %u", GetNumberOfLeafNodeSOA());

  return true;
}

void Hybrid::Thread_BuildExtendLeafNodeOnCPU(ul current_offset, ul parent_offset, 
                                             ui number_of_node, ui tid, ui number_of_threads) {
  node::Node_SOA* current_node;
  node::Node_SOA* parent_node;

  for(ui range(node_offset, tid, number_of_node, number_of_threads)) {
    current_node = node_soa_ptr+current_offset+node_offset;
    parent_node = node_soa_ptr+parent_offset+(ul)(node_offset/GetNumberOfDegrees());

    parent_node->SetChildOffset(node_offset%GetNumberOfDegrees(), 
                                (ll)current_node-(ll)parent_node);

    parent_node->SetIndex(node_offset%GetNumberOfDegrees(), current_node->GetLastIndex());

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

      for( ui range(thread_itr, 0, GetNumberOfDegrees())) {
        if( thread_itr < current_node->GetBranchCount()){
          lower_boundary[ thread_itr ] = current_node->GetBranchPoint(thread_itr, dim);
          upper_boundary[ thread_itr ] = current_node->GetBranchPoint(thread_itr, high_dim);
        } else {
          lower_boundary[ thread_itr ] = 1.0f;
          upper_boundary[ thread_itr ] = 0.0f;
        }
      }

      FindMinOnCPU(lower_boundary, GetNumberOfDegrees());
      FindMaxOnCPU(upper_boundary, GetNumberOfDegrees());

      parent_node->SetBranchPoint( (node_offset % GetNumberOfDegrees()), lower_boundary[0], dim);
      parent_node->SetBranchPoint( (node_offset % GetNumberOfDegrees()), upper_boundary[0], high_dim);
    }
  }

  //last node in each level
  if(  number_of_node % GetNumberOfDegrees() ){
    parent_node = node_soa_ptr + current_offset - 1;
    if( number_of_node < GetNumberOfDegrees() ) {
      parent_node->SetBranchCount(number_of_node);
    }else{
      parent_node->SetBranchCount(number_of_node%GetNumberOfDegrees());
    }
  }
}

bool Hybrid::BuildExtendLeafNodeOnCPU() {

  const size_t number_of_threads = std::thread::hardware_concurrency();

  // parallel for loop using c++ std 11 
  {
    std::vector<std::thread> threads;
    ul current_offset = GetNumberOfExtendLeafNodeSOA();
    ul parent_offset = 0;

    for (ui range(thread_itr, 0, number_of_threads)) {
      threads.push_back(std::thread(&Hybrid::Thread_BuildExtendLeafNodeOnCPU, 
                        this, current_offset, parent_offset, 
                        GetNumberOfLeafNodeSOA(), thread_itr, number_of_threads));
    }
    //Join the threads with the main thread
    for(auto &thread : threads){
      thread.join();
    }
    threads.clear();
  }

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
  recorder.TimeRecordStart();

  //===--------------------------------------------------------------------===//
  // Node counts
  //===--------------------------------------------------------------------===//
  // read total node count
  fread(&total_node_count, sizeof(ui), 1, index_file);

  // read node soa count
  fread(&extend_leaf_node_soa_count, sizeof(ui), 1, index_file);

  // read node soa count
  fread(&leaf_node_soa_count, sizeof(ui), 1, index_file);

  //===--------------------------------------------------------------------===//
  // Internal nodes
  //===--------------------------------------------------------------------===//
  node_ptr = new node::Node[total_node_count];
  fread(node_ptr, sizeof(node::Node), total_node_count, index_file);


  //===--------------------------------------------------------------------===//
  // Extend & leaf nodes
  //===--------------------------------------------------------------------===//
  node_soa_ptr = new node::Node_SOA[GetNumberOfNodeSOA()];
  fread(node_soa_ptr, sizeof(node::Node_SOA), GetNumberOfNodeSOA(), index_file);

  fclose(index_file);

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Done, time = %.6fs", elapsed_time/1000.0f);

  return true;
}

bool Hybrid::DumpToFile(std::string index_name) {
  auto& recorder = evaluator::Recorder::GetInstance();

  LOG_INFO("Dump an index into file (%s)...", index_name.c_str());

  recorder.TimeRecordStart();
  // NOTE :: Use fwrite since it is fast
  FILE* index_file;
  index_file = fopen(index_name.c_str(),"wb");

  //===--------------------------------------------------------------------===//
  // Node counts
  //===--------------------------------------------------------------------===//
  // write total node count
  fwrite(&total_node_count, sizeof(ui), 1, index_file);

  // write extend leaf node soa count 
  fwrite(&extend_leaf_node_soa_count, sizeof(ui), 1, index_file);

  // write leaf node soa count 
  fwrite(&leaf_node_soa_count, sizeof(ui), 1, index_file);


  //===--------------------------------------------------------------------===//
  // Internal nodes
  //===--------------------------------------------------------------------===//

  // Unlike dump function in MPHR class, we use the queue structure to dump the
  // tree onto an index file since the nodes are allocated here and there in a
  // Top-Down fashion
  std::queue<node::Node*> bfs_queue;
  std::vector<ll> original_child_offset; // for backup

  // push the root node
  bfs_queue.emplace(node_ptr);

  // if the queue is not empty,
  while(!bfs_queue.empty()) {
    // pop the first element 
    node::Node* node = bfs_queue.front();
    bfs_queue.pop();

    // NOTE : Backup the child offsets in order to recover node's child offset later
    // I believe accessing memory is faster than accesing disk,
    // I don't use another fwrite for this job.
    if( node->GetNodeType() == NODE_TYPE_INTERNAL) {
      for(ui range(child_itr, 0, node->GetBranchCount())) {
        node::Node* child_node = node->GetBranchChildNode(child_itr);
        bfs_queue.emplace(child_node);

        // backup current child offset
        original_child_offset.emplace_back(node->GetBranchChildOffset(child_itr));

        // reassign child offset
        ll child_offset = (ll)bfs_queue.size()*(ll)sizeof(node::Node);
        node->SetBranchChildOffset(child_itr, child_offset);
      }
    }

    // write an internal node on disk
    fwrite(node, sizeof(node::Node), 1, index_file);

    // Recover child offset
    if( node->GetNodeType() == NODE_TYPE_INTERNAL) {
      for(ui range(child_itr, 0, node->GetBranchCount())) {
        // reassign child offset
        node->SetBranchChildOffset(child_itr, original_child_offset[child_itr]);
      }
    }
    original_child_offset.clear();
  }

  //===--------------------------------------------------------------------===//
  // Extend & leaf nodes
  //===--------------------------------------------------------------------===//
  fwrite(node_soa_ptr, sizeof(node::Node_SOA), GetNumberOfNodeSOA(), index_file);
  fclose(index_file);

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Done, time = %.6fs", elapsed_time/1000.0f);
  return true;
}

void Hybrid::SetNumberOfNodeSOA(ui number_of_data) {
  leaf_node_soa_count = std::ceil((float)number_of_data/(float)GetNumberOfDegrees());
  assert(leaf_node_soa_count);

  extend_leaf_node_soa_count = std::ceil((float)leaf_node_soa_count/(float)GetNumberOfDegrees());
  assert(extend_leaf_node_soa_count);
}

ui Hybrid::GetNumberOfNodeSOA() const {
  assert(GetNumberOfLeafNodeSOA());
  assert(GetNumberOfExtendLeafNodeSOA());
  return GetNumberOfLeafNodeSOA() + GetNumberOfExtendLeafNodeSOA();
}

ui Hybrid::GetNumberOfLeafNodeSOA() const {
  assert(leaf_node_soa_count);
  return leaf_node_soa_count;
}

ui Hybrid::GetNumberOfExtendLeafNodeSOA() const {
  assert(extend_leaf_node_soa_count);
  return extend_leaf_node_soa_count;
}

int Hybrid::Search(std::shared_ptr<io::DataSet> query_data_set, 
                   ui number_of_search, ui number_of_repeat){

  auto& recorder = evaluator::Recorder::GetInstance();

  //===--------------------------------------------------------------------===//
  // Read Query 
  //===--------------------------------------------------------------------===//
  auto query = query_data_set->GetPoints();
  auto d_query = query_data_set->GetDeviceQuery(number_of_search);

  for(ui range(repeat_itr, 0, number_of_repeat)) {
    LOG_INFO("#%u) Evaluation", repeat_itr+1);
    //===--------------------------------------------------------------------===//
    // Prepare Hit & Node Visit Variables for an evaluation
    //===--------------------------------------------------------------------===//
    ui h_hit[GetNumberOfBlocks()];
    ui h_node_visit_count[GetNumberOfBlocks()];

    ui total_hit = 0;
    ui total_jump_count = 0;
    ui total_node_visit_count_cpu = 0;
    ui total_node_visit_count_gpu = 0;

    ui* d_hit;
    cudaErrCheck(cudaMalloc((void**) &d_hit, sizeof(ui)*GetNumberOfBlocks()));
    ui* d_node_visit_count;
    cudaErrCheck(cudaMalloc((void**) &d_node_visit_count, sizeof(ui)*GetNumberOfBlocks()));

    // initialize hit and node visit variables to zero
    global_SetHitCount<<<1,GetNumberOfBlocks()>>>(0);
    cudaDeviceSynchronize();

    //===--------------------------------------------------------------------===//
    // Prepare Multi-thread Query Processing
    //===--------------------------------------------------------------------===//
    ui number_of_blocks_per_cpu = GetNumberOfBlocks()/number_of_cpu_threads;
    // chunk size should be equal or larger than number of blocks per cpu
    // otherwise, just wasting GPU resources.
    assert(chunk_size >= number_of_blocks_per_cpu);

    std::vector<std::thread> threads;
    ui thread_jump_count[number_of_cpu_threads];
    ui thread_node_visit_count_cpu[number_of_cpu_threads];

    //===--------------------------------------------------------------------===//
    // Collect Start Node Index in Advance
    //===--------------------------------------------------------------------===//
    /*
    // NOTE : Collect start node index in advance to measure GPU kernel launching time
    thread_start_node_index.resize(number_of_cpu_threads);
    // parallel for loop using c++ std 11 
    {
      auto chunk_size = number_of_search/number_of_cpu_threads;
      auto start_offset = 0 ;
      auto end_offset = start_offset + chunk_size + number_of_search%number_of_cpu_threads;

      for (ui range(thread_itr, 0, number_of_cpu_threads)) {
        threads.push_back(std::thread(&Hybrid::Thread_CollectStartNodeIndex, this, 
              std::ref(query), std::ref(thread_start_node_index[thread_itr]),
              start_offset, end_offset));
        start_offset = end_offset;
        end_offset += chunk_size;
      }

      //Join the threads with the main thread
      for(auto &thread : threads){
        thread.join();
      }
    }
    threads.clear();
    */

    //===--------------------------------------------------------------------===//
    // Execute Search Function
    //===--------------------------------------------------------------------===//
    recorder.TimeRecordStart();

    // parallel for loop using c++ std 11 
    {
      auto search_chunk_size = number_of_search/number_of_cpu_threads;
      auto start_offset = 0 ;
      auto end_offset = start_offset + search_chunk_size + number_of_search%number_of_cpu_threads;

      for (ui range(thread_itr, 0, number_of_cpu_threads)) {
        threads.push_back(std::thread(&Hybrid::Thread_Search, this, 
              std::ref(query), d_query, 
              thread_itr, number_of_blocks_per_cpu, 
              std::ref(thread_jump_count[thread_itr]), 
              std::ref(thread_node_visit_count_cpu[thread_itr]),
              start_offset, end_offset));

        start_offset = end_offset;
        end_offset += search_chunk_size;
      }

      //Join the threads with the main thread
      for(auto &thread : threads){
        thread.join();
      }

      for(ui range(thread_itr, 0, number_of_cpu_threads)) {
        total_jump_count += thread_jump_count[thread_itr];
        total_node_visit_count_cpu += thread_node_visit_count_cpu[thread_itr];
      }
    }

    LOG_INFO("Avg. Jump Count %f", total_jump_count/(float)number_of_search);

    // A problem with using host-device synchronization points, such as
    // cudaDeviceSynchronize(), is that they stall the GPU pipeline
    cudaDeviceSynchronize();

    global_GetHitCount<<<1,GetNumberOfBlocks()>>>(d_hit, d_node_visit_count);
    cudaMemcpy(h_hit, d_hit, sizeof(ui)*GetNumberOfBlocks(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_node_visit_count, d_node_visit_count, sizeof(ui)*GetNumberOfBlocks(), 
        cudaMemcpyDeviceToHost);

    for(ui range(i, 0, GetNumberOfBlocks())) {
      total_hit += h_hit[i];
      total_node_visit_count_gpu += h_node_visit_count[i];
    }

    auto elapsed_time = recorder.TimeRecordEnd();
    LOG_INFO("%zu threads processing queries concurrently", number_of_cpu_threads);
    LOG_INFO("Search Time on the GPU = %.6fms", elapsed_time);

    //===--------------------------------------------------------------------===//
    // Show Results
    //===--------------------------------------------------------------------===//
    LOG_INFO("Hit : %u", total_hit);
    LOG_INFO("Node visit count on CPU : %u", total_node_visit_count_cpu);
    LOG_INFO("Node visit count on GPU : %u\n\n", total_node_visit_count_gpu);
  }
}

void Hybrid::Thread_CollectStartNodeIndex(std::vector<Point>& query,
                                          std::queue<ll> &start_node_indice,
                                          ui start_offset, ui end_offset){
  ui node_visit_count = 0;

  auto number_of_nodes = GetNumberOfLeafNodeSOA();
  if(scan_type == SCAN_TYPE_EXTENDLEAF) {
    number_of_nodes = GetNumberOfExtendLeafNodeSOA();
  }

  for(ui range(query_itr, start_offset, end_offset)) {
    ui query_offset = query_itr*GetNumberOfDims()*2;
    ll visited_leafIndex = 0;

    while(1) {
      //===--------------------------------------------------------------------===//
      // Traversal Internal Nodes on CPU
      //===--------------------------------------------------------------------===//
      auto start_node_index = TraverseInternalNodes(node_ptr, &query[query_offset],
                                               visited_leafIndex, &node_visit_count);
        
      // collect start node index
      start_node_indice.emplace(start_node_index);

      // no more overlapping internal nodes, terminate current query
      if( start_node_index == 0) {
        break;
      }

      auto start_node_offset = (start_node_index-1)/GetNumberOfDegrees(); 
      if(scan_type == SCAN_TYPE_EXTENDLEAF)  {
        start_node_offset /= GetNumberOfDegrees(); 
      }

      // resize chunk_size if the sum of start node offset and chunk size is
      // larger than number of leaf nodes
      if(start_node_offset+chunk_size > number_of_nodes) {
        chunk_size = number_of_nodes - start_node_offset;
      }

      visited_leafIndex = (start_node_offset+chunk_size)*GetNumberOfDegrees();

      if(scan_type == SCAN_TYPE_EXTENDLEAF) {
        visited_leafIndex *= GetNumberOfDegrees();
      }
    }
  }
}

ll Hybrid::GetNextStartNodeIndex(ui tid) {
  auto start_node_index =  thread_start_node_index[tid].front();
  thread_start_node_index[tid].pop();
  return start_node_index;
}

void Hybrid::Thread_Search(std::vector<Point>& query, Point* d_query, ui tid,
                           ui number_of_blocks_per_cpu, ui& jump_count, 
                           ui& node_visit_count, ui start_offset, ui end_offset) {

  // Get Chunk Manager and initialize it
  //auto& chunk_manager = manager::ChunkManager::GetInstance();

  ui bid_offset = tid*number_of_blocks_per_cpu;
  jump_count = 0;
  node_visit_count = 0;

  ll start_node_index;
  ll start_node_offset;
  ui query_offset = start_offset*GetNumberOfDims()*2;

  auto number_of_nodes = GetNumberOfLeafNodeSOA();
  if(scan_type == SCAN_TYPE_EXTENDLEAF) {
    number_of_nodes = GetNumberOfExtendLeafNodeSOA();
  }

  for(ui range(query_itr, start_offset, end_offset)) {
    ll visited_leafIndex = 0;

    while(1) {

      //===--------------------------------------------------------------------===//
      // Traversal Internal Nodes on CPU
      //===--------------------------------------------------------------------===//
      start_node_index = TraverseInternalNodes(node_ptr, &query[query_offset], visited_leafIndex, &node_visit_count);
      //start_node_index = GetNextStartNodeIndex(tid);

      // no more overlapping internal nodes, terminate current query
      if( start_node_index == 0) {
        break;
      }

      start_node_offset = (start_node_index-1)/GetNumberOfDegrees(); 
      if(scan_type == SCAN_TYPE_EXTENDLEAF)  {
        start_node_offset /= GetNumberOfDegrees(); 
      }

      // resize chunk_size if the sum of start node offset and chunk size is
      // larger than number of leaf nodes
      if(start_node_offset+chunk_size > number_of_nodes) {
        chunk_size = number_of_nodes - start_node_offset;
      }

      //===--------------------------------------------------------------------===//
      // Parallel Scanning Leaf Nodes on the GPU 
      //===--------------------------------------------------------------------===//
      if(scan_type == SCAN_TYPE_LEAF) {
        //chunk_manager.CopyNode(node_soa_ptr+GetNumberOfExtendLeafNodeSOA(), 
        //                       start_node_offset, chunk_size);
        global_ParallelScan_Leafnodes<<<number_of_blocks_per_cpu,GetNumberOfThreads()>>>
                                      (&d_query[query_offset], start_node_offset, chunk_size,
                                       bid_offset, number_of_blocks_per_cpu );
      } else if(scan_type == SCAN_TYPE_EXTENDLEAF) {
        global_ParallelScan_ExtendLeafnodes<<<number_of_blocks_per_cpu,GetNumberOfThreads()>>>
                                            (&d_query[query_offset], start_node_offset, chunk_size,
                                            bid_offset, number_of_blocks_per_cpu );
      }
      visited_leafIndex = (start_node_offset+chunk_size)*GetNumberOfDegrees();

      if(scan_type == SCAN_TYPE_EXTENDLEAF) {
        visited_leafIndex *= GetNumberOfDegrees();
      }

      jump_count++;
    }
    query_offset += GetNumberOfDims()*2;
  }
}

void Hybrid::SetChunkSize(ui _chunk_size){
  chunk_size = _chunk_size;
}

void Hybrid::SetScanType(ScanType _scan_type){
  scan_type = _scan_type;
  assert(scan_type);
}

void Hybrid::SetNumberOfCPUThreads(ui _number_of_cpu_threads){
  number_of_cpu_threads = _number_of_cpu_threads;
  assert(number_of_cuda_blocks/number_of_cpu_threads>0);
}

void Hybrid::SetNumberOfCUDABlocks(ui _number_of_cuda_blocks){
  number_of_cuda_blocks = _number_of_cuda_blocks;
  assert(number_of_cuda_blocks);
}

ll Hybrid::TraverseInternalNodes(node::Node *node_ptr, Point* query, 
                                 ll visited_leafIndex, ui *node_visit_count) {

  ll start_node_index=0;
  (*node_visit_count)++;

  // internal nodes
  if(node_ptr->GetNodeType() == NODE_TYPE_INTERNAL ) {
    for(ui range(branch_itr, 0, node_ptr->GetBranchCount())) {
      if( node_ptr->GetBranchIndex(branch_itr) > visited_leafIndex && 
          node_ptr->IsOverlap(query, branch_itr)) {
        start_node_index=TraverseInternalNodes(node_ptr->GetBranchChildNode(branch_itr), 
                                            query, visited_leafIndex, node_visit_count);

        if(start_node_index > 0) break;
      }
    }
  } // leaf nodes
  else {
    if(scan_type == SCAN_TYPE_LEAF) {
      for(ui range(branch_itr, 0, node_ptr->GetBranchCount())) {
        if( node_ptr->GetBranchIndex(branch_itr) > visited_leafIndex ) {
          start_node_index = node_ptr->GetBranchIndex(branch_itr);
          break;
        }
      }
    } else if(scan_type == SCAN_TYPE_EXTENDLEAF) {
      start_node_index = node_ptr->GetBranchIndex(0);
      if( start_node_index <= visited_leafIndex) {
        start_node_index = visited_leafIndex+1;
      }
    }
  }
  return start_node_index;
}

//===--------------------------------------------------------------------===//
// Cuda Variable & Function 
//===--------------------------------------------------------------------===//

__device__ ui g_hit[GetNumberOfMAXBlocks()]; 
__device__ ui g_node_visit_count[GetNumberOfMAXBlocks()]; 

__global__
void global_SetHitCount(ui init_value) {
  int tid = threadIdx.x;

  g_hit[tid] = init_value;
  g_node_visit_count[tid] = init_value;
}

__global__
void global_GetHitCount(ui* hit, ui* node_visit_count) {
  int tid = threadIdx.x;

  hit[tid] = g_hit[tid];
  node_visit_count[tid] = g_node_visit_count[tid];
}


//===--------------------------------------------------------------------===//
// Scan Type Leaf
//===--------------------------------------------------------------------===//
__global__ 
void global_ParallelScan_Leafnodes(Point* _query, ll start_node_offset, 
                                       ui chunk_size, ui bid_offset, 
                                       ui number_of_blocks_per_cpu) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  __shared__ ui t_hit[GetNumberOfThreads()]; 
  __shared__ Point query[GetNumberOfDims()*2];

  if(tid < GetNumberOfDims()*2) {
    query[tid] = _query[tid];
  }

  t_hit[tid] = 0;

  node::Node_SOA* node_soa_ptr = manager::g_node_soa_ptr/*first leaf node*/ + start_node_offset + bid;
  __syncthreads();

  //===--------------------------------------------------------------------===//
  // Leaf Nodes
  //===--------------------------------------------------------------------===//

  for(ui range(node_itr, bid, chunk_size, number_of_blocks_per_cpu)) {

    MasterThreadOnly {
      g_node_visit_count[bid+bid_offset]++;
    }

    if(tid < node_soa_ptr->GetBranchCount() &&
        node_soa_ptr->IsOverlap(query, tid)) {
      t_hit[tid]++;
    }
    __syncthreads();

    node_soa_ptr+=number_of_blocks_per_cpu;
  }
  __syncthreads();

  //===--------------------------------------------------------------------===//
  // Parallel Reduction 
  //===--------------------------------------------------------------------===//
  ParallelReduction(t_hit, GetNumberOfThreads());

  MasterThreadOnly {
    g_hit[bid+bid_offset] += t_hit[0] + t_hit[1];
  }
}


//===--------------------------------------------------------------------===//
// Scan Type Extend Leaf
//===--------------------------------------------------------------------===//
__global__ 
void global_ParallelScan_ExtendLeafnodes(Point* _query, ll start_node_offset, 
                                             ui chunk_size, ui bid_offset, 
                                             ui number_of_blocks_per_cpu) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  __shared__ bool child_overlap[GetNumberOfThreads()]; 
  __shared__ ui t_hit[GetNumberOfThreads()]; 
  __shared__ Point query[GetNumberOfDims()*2];

  if(tid < GetNumberOfDims()*2) {
    query[tid] = _query[tid];
  }

  t_hit[tid] = 0;

  // start from the first extend leaf node
  node::Node_SOA* node_soa_ptr = manager::g_node_soa_ptr + start_node_offset + bid;
  __syncthreads();

  for(ui range(node_itr, bid, chunk_size, number_of_blocks_per_cpu)) {
    MasterThreadOnly {
      g_node_visit_count[bid+bid_offset]++;
    }

    //===--------------------------------------------------------------------===//
    // Extend Leaf Nodes
    //===--------------------------------------------------------------------===//
    if(tid < node_soa_ptr->GetBranchCount() &&
        node_soa_ptr->IsOverlap(query, tid)) {
      child_overlap[tid] = true;
    } else {
      child_overlap[tid] = false;
    }
    __syncthreads();

    //===--------------------------------------------------------------------===//
    // Leaf Nodes
    //===--------------------------------------------------------------------===//
    for( ui range(child_itr, 0, node_soa_ptr->GetBranchCount())) {
      if( child_overlap[child_itr]) {

        MasterThreadOnly {
          g_node_visit_count[bid+bid_offset]++;
        }

        auto child_node = node_soa_ptr->GetChildNode(child_itr);

        if(tid < child_node->GetBranchCount() &&
            child_node->IsOverlap(query, tid)) {
          t_hit[tid]++;
        }
        __syncthreads();
      }
    }
    node_soa_ptr += number_of_blocks_per_cpu;
  }
  __syncthreads();

  //===--------------------------------------------------------------------===//
  // Parallel Reduction 
  //===--------------------------------------------------------------------===//
  ParallelReduction(t_hit, GetNumberOfThreads());

  MasterThreadOnly {
      g_hit[bid+bid_offset] += t_hit[0] + t_hit[1];
  }
}

} // End of tree namespace
} // End of ursus namespace

