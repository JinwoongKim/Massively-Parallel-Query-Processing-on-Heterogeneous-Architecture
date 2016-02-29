#include "transformer/transformer.h"

#include "common/macro.h"
#include "node/branch.h"

// XXX for debugging
#include "common/logger.h"

namespace ursus {
namespace transformer {

/**
 * @brief Transform the Node into G_Node
 * @param node node pointer
 * @param number_of_nodes
 * @return G_Node pointer if success otherwise nullptr
 */
node::G_Node* Transformer::Transform(node::Node* node,
                                        ui number_of_nodes) {

  node::G_Node* g_node = new node::G_Node[number_of_nodes];

  for(ui range(node_itr, 0, number_of_nodes)) {
    auto number_of_branches = node[node_itr].GetBranchCount();
    g_node[node_itr].SetBranchCount(number_of_branches);

    for(ui range(branch_itr, 0, number_of_branches)) {
      auto branch = node[node_itr].GetBranch(branch_itr);

      auto points = branch.GetPoints();
      auto index = branch.GetIndex();
      auto child = branch.GetChild();


      // set points in G_Node
      for(ui range(dim_itr, 0, GetNumberOfDims()*2)) {
        auto offset = dim_itr*GetNumberOfDegrees()+branch_itr;
        g_node[node_itr].SetPoint(offset, points[dim_itr]);
      }

      // set the index
      g_node[node_itr].SetIndex(branch_itr, index);

      LOG_INFO("node %p", node);
      LOG_INFO("g_node %p", g_node);
      LOG_INFO("node[%u] %p",node_itr, &node[node_itr]);
      LOG_INFO("child %p", child);
      LOG_INFO("node size %zu", sizeof(node::Node));

      // get the child node offset 
      auto child_offset = (child - &node[node_itr])/sizeof(node::Node);
      LOG_INFO("child_offset %zu",child_offset);

      // child ptr for G_Node
      auto g_node_child = g_node+child_offset;
      LOG_INFO("g_node_child %p",g_node_child);

      g_node[node_itr].SetChild(branch_itr, g_node_child);
    }

    // node type 
    g_node[node_itr].SetNodeType(node[node_itr].GetNodeType());

    // node level
    g_node[node_itr].SetLevel(node[node_itr].GetLevel());
  }

  return g_node;
}

} // End of transformer namespace
} // End of ursus namespace
