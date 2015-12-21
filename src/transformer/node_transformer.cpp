#include "transformer/node_transformer.h"

#include "common/macro.h"

namespace ursus {
namespace transformer {

/**
 * @brief Transform the Node into G_Node
 * @param node_ptr node pointer
 * @param number_of_nodes
 * @return G_Node pointer if success otherwise nullptr
 */
node::G_Node_Ptr Node_Transformer::Transform(node::Node_Ptr node_ptr,
                                             ui number_of_nodes) {

  node::G_Node_Ptr g_node_ptr = new node::G_Node[number_of_nodes];

  for(ui range(node_itr, 0, number_of_nodes)) {
    auto number_of_branches = node_ptr[node_itr].GetBranchCount();
    g_node_ptr[node_itr].SetBranchCount(number_of_branches);

    for(ui range(branch_itr, 0, number_of_branches)) {
      auto branch = node_ptr[node_itr].GetBranch(branch_itr);

      auto points = branch.GetPoints();
      auto index = branch.GetIndex();
      auto child = branch.GetChild();


      // set points in G_Node
      for(ui range(dim_itr, 0, GetNumberOfDims()*2)) {
        auto offset = dim_itr*GetNumberOfDegress()+branch_itr;
        g_node_ptr[node_itr].SetPoint(offset, points[dim_itr]);
      }

      // set the index
      g_node_ptr[node_itr].SetIndex(branch_itr, index);

      LOG_INFO("node %p", node_ptr);
      LOG_INFO("g_node %p", g_node_ptr);
      LOG_INFO("node[%d] %p",node_itr, node_ptr[node_itr]);
      LOG_INFO("child %p", child);
      LOG_INFO("node size %d", sizeof(node::Node));

      // get the child node offset 
      auto child_offset = (child - node_ptr[node_itr])/sizeof(node::Node);
      LOG_INFO("child_offset %d",child_offset);

      // child ptr for G_Node
      auto g_node_child = g_node_ptr+child_offset;
      LOG_INFO("g_node_child %p",g_node_child);

      g_node_ptr[node_itr].SetChild(branch_itr, g_node_child);
    }

    // node type 
    g_node_ptr.SetNodeType(node_ptr->GetNodeType());

    // node level
    g_node_ptr.SetLevel(node_ptr->GetLevel());
  }

  return g_node_ptr;
}

} // End of transformer namespace
} // End of ursus namespace
