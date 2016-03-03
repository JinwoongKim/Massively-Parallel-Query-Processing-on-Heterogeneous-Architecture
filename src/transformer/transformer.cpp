#include "transformer/transformer.h"

#include "common/macro.h"
#include "common/logger.h"
#include "evaluator/recorder.h"
#include "node/branch.h"


namespace ursus {
namespace transformer {

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

  for(ui range(node_itr, 0, number_of_nodes)) {
    auto number_of_branches = node[node_itr].GetBranchCount();
    node_soa[node_itr].SetBranchCount(number_of_branches);

    for(ui range(branch_itr, 0, number_of_branches)) {
      auto branch = node[node_itr].GetBranch(branch_itr);

      auto points = branch.GetPoints();
      auto index = branch.GetIndex();
      auto child_offset = branch.GetChildOffset();


      // set points in Node_SOA
      for(ui range(dim_itr, 0, GetNumberOfDims()*2)) {
        auto offset = dim_itr*GetNumberOfDegrees()+branch_itr;
        node_soa[node_itr].SetPoint(offset, points[dim_itr]);
      }

      // set the index
      node_soa[node_itr].SetIndex(branch_itr, index);
      node_soa[node_itr].SetChildOffset(branch_itr, child_offset);
    }

    // node type 
    node_soa[node_itr].SetNodeType(node[node_itr].GetNodeType());

    // node level
    node_soa[node_itr].SetLevel(node[node_itr].GetLevel());
  }

  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Transform Time on the GPU = %.6fs", elapsed_time/1000.0f);

  return node_soa;
}

} // End of transformer namespace
} // End of ursus namespace
