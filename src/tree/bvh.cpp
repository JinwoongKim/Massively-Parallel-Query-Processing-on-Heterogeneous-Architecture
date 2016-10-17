#include "tree/bvh.h"

#include "common/macro.h"
#include "common/logger.h"
#include "evaluator/recorder.h"
#include "sort/sorter.h"

#include <cassert>
#include <queue>
#include <thread>
#include <algorithm>

namespace ursus {
namespace tree {

BVH::BVH() { 
  tree_type = TREE_TYPE_BVH;
}

/**
 * @brief build trees
 * @param input_data_set 
 * @return true if success to build otherwise false
 */
bool BVH::Build(std::shared_ptr<io::DataSet> input_data_set){

  LOG_INFO("Build BVH");
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

    //===--------------------------------------------------------------------===//
    // Build the internal nodes in a top-down fashion 
    //===--------------------------------------------------------------------===//
    ret = Top_Down(branches, GetTreeType()); 
    //ret = Top_Down(branches); 
    assert(ret);

    // Dump an index to the file
    DumpToFile(index_name);
  }

  return true;
}

bool BVH::DumpFromFile(std::string index_name) {

  FILE* index_file = OpenIndexFile(index_name);
  if(index_file == nullptr) {
    return false;
  }
  auto& recorder = evaluator::Recorder::GetInstance();
  recorder.TimeRecordStart();

  //===--------------------------------------------------------------------===//
  // Node counts
  //===--------------------------------------------------------------------===//
  // read total node count
  fread(&host_node_count, sizeof(ui), 1, index_file);

  //===--------------------------------------------------------------------===//
  // Internal nodes
  //===--------------------------------------------------------------------===//
  node_ptr = new node::Node[host_node_count];
  fread(node_ptr, sizeof(node::Node), host_node_count, index_file);

  fclose(index_file);

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Done, time = %.6fs", elapsed_time/1000.0f);

  return true;
}

bool BVH::DumpToFile(std::string index_name) {
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
  fwrite(&host_node_count, sizeof(ui), 1, index_file);

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

  fclose(index_file);

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Done, time = %.6fs", elapsed_time/1000.0f);
  return true;
}

int BVH::Search(std::shared_ptr<io::DataSet> query_data_set, 
                   ui number_of_search, ui number_of_repeat){

  auto& recorder = evaluator::Recorder::GetInstance();

  //===--------------------------------------------------------------------===//
  // Read Query 
  //===--------------------------------------------------------------------===//
  auto query = query_data_set->GetPoints();
  
  for(ui range(repeat_itr, 0, number_of_repeat)) {
    if( number_of_repeat > 1){
      LOG_INFO("#%u) Evaluation", repeat_itr+1);
    }
    //===--------------------------------------------------------------------===//
    // Prepare Multi-thread Query Processing
    //===--------------------------------------------------------------------===//
    std::vector<std::thread> threads;
    ui thread_hit[number_of_cpu_threads];
    ui thread_node_visit_count[number_of_cpu_threads];
    
    ui total_hit=0;
    ui total_node_visit_count=0;
  
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
        threads.push_back(std::thread(&BVH::Thread_Search, this, 
                          std::ref(query), thread_itr,  
                          std::ref(thread_hit[thread_itr]), 
                          std::ref(thread_node_visit_count[thread_itr]),
                          start_offset, end_offset));
  
        start_offset = end_offset;
        end_offset += search_chunk_size;
      }
  
      //Join the threads with the main thread
      for(auto &thread : threads){
        thread.join();
      }
  
      for(ui range(thread_itr, 0, number_of_cpu_threads)) {
        total_hit += thread_hit[thread_itr];
        total_node_visit_count += thread_node_visit_count[thread_itr];
      }
    }
  
    auto elapsed_time = recorder.TimeRecordEnd();
    LOG_INFO("%u threads processing queries concurrently", number_of_cpu_threads);
  
    //===--------------------------------------------------------------------===//
    // Show Results
    //===--------------------------------------------------------------------===//
    LOG_INFO("Hit : %u", total_hit);
    LOG_INFO("Avg. Search Time on the CPU (ms)\n%.6f", elapsed_time/(float)number_of_search);
    LOG_INFO("Total Search Time on the CPU (ms)%.6f", elapsed_time);
    LOG_INFO("Avg. Node visit count : %f", total_node_visit_count/(float)number_of_search);
    LOG_INFO("Total Node visit count : %u", total_node_visit_count);
    LOG_INFO("\n");
  }
  return 1;
}

void BVH::SetNumberOfCPUThreads(ui _number_of_cpu_threads) {
  number_of_cpu_threads = _number_of_cpu_threads;
  assert(number_of_cpu_threads);
}

void BVH::Thread_Search(std::vector<Point>& query, ui tid,
                           ui& hit, ui& node_visit_count, ui start_offset, ui end_offset) {
  hit = 0;
  node_visit_count = 0;

  ui query_offset = start_offset*GetNumberOfDims()*2;

  for(ui range(query_itr, start_offset, end_offset)) {
    hit += TraverseInternalNodes(node_ptr, &query[query_offset], &node_visit_count);
    query_offset += GetNumberOfDims()*2;
  }
}

ui BVH::TraverseInternalNodes(node::Node *node_ptr, Point* query, 
                                 ui *node_visit_count) {
  ui hit = 0;
  (*node_visit_count)++;

  // internal nodes
  if(node_ptr->GetNodeType() == NODE_TYPE_INTERNAL ) {
    for(ui range(branch_itr, 0, node_ptr->GetBranchCount())) {
      if( node_ptr->IsOverlap(query, branch_itr)) {
        hit += TraverseInternalNodes(node_ptr->GetBranchChildNode(branch_itr), 
                                     query, node_visit_count);
      }
    }
  } // leaf nodes
  else {
    for(ui range(branch_itr, 0, node_ptr->GetBranchCount())) {
      if( node_ptr->IsOverlap(query, branch_itr)) {
        hit++;
      }
    }
  }
  return hit;
}

} // End of tree namespace
} // End of ursus namespace
