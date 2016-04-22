#include "transformer/transformer.h"

#include "common/macro.h"
#include "common/logger.h"
#include "evaluator/recorder.h"
#include "node/branch.h"

#include <thread>

namespace ursus {
namespace transformer {

void Thread_Transform(node::Node *node, node::Node_SOA *node_soa,
                      ui start_offset, ui end_offset) {

  for(ui range(node_offset, start_offset, end_offset)) {
    auto number_of_branches = node[node_offset].GetBranchCount();
    node_soa[node_offset].SetBranchCount(number_of_branches);

    for(ui range(branch_itr, 0, number_of_branches)) {
      auto branch = node[node_offset].GetBranch(branch_itr);

      auto points = branch.GetPoints();
      auto index = branch.GetIndex();
      auto child_offset = branch.GetChildOffset();

      // set points in Node_SOA
      for(ui range(dim_itr, 0, GetNumberOfDims()*2)) {
        auto offset = dim_itr*GetNumberOfDegrees()+branch_itr;
        node_soa[node_offset].SetPoint(offset, points[dim_itr]);
      }

      // set the index
      node_soa[node_offset].SetIndex(branch_itr, index);
      node_soa[node_offset].SetChildOffset(branch_itr, child_offset);
    }

    // node type 
    node_soa[node_offset].SetNodeType(node[node_offset].GetNodeType());

    // node level
    node_soa[node_offset].SetLevel(node[node_offset].GetLevel());
  }
}

/**
 * @brief Transform the Node into Node_SOA
 * @param node node pointer
 * @param number_of_nodes
 * @return Node_SOA pointer if success otherwise nullptr
 */
node::Node_SOA* Transformer::Transform(node::Node* node,
                                        ui number_of_nodes) {
  auto& recorder = evaluator::Recorder::GetInstance();
  recorder.TimeRecordStart();

  node::Node_SOA* node_soa = new node::Node_SOA[number_of_nodes];

  const size_t number_of_threads = std::thread::hardware_concurrency();

  // parallel for loop using c++ std 11 
  {
    std::vector<std::thread> threads;

    auto chunk_size = number_of_nodes/number_of_threads;
    auto start_offset = 0 ;
    auto end_offset = start_offset + chunk_size + number_of_nodes%number_of_threads;
      
    //Launch a group of threads
    for (ui range(thread_itr, 0, number_of_threads)) {
      threads.push_back(std::thread(Thread_Transform, std::ref(node), 
                        std::ref(node_soa), start_offset, end_offset));

      start_offset = end_offset;
      end_offset += chunk_size;
    }

    //Join the threads with the main thread
    for(auto &thread : threads){
      thread.join();
    }
  }

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Transform Time on the CPU (%zu threads) = %.6fs", number_of_threads, elapsed_time/1000.0f);

  return node_soa;
}

} // End of transformer namespace
} // End of ursus namespace
