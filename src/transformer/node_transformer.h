#pragma once

#include "node/node.h"
#include "node/g_node.h"

namespace ursus {
namespace transformer {

class Node_Transformer{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  Node_Transformer(const Node_Transformer &) = delete;
  Node_Transformer &operator=(const Node_Transformer &) = delete;
  Node_Transformer(Node_Transformer &&) = delete;
  Node_Transformer &operator=(Node_Transformer &&) = delete;

  // global singleton
  static Node_Transformer& GetInstance(void);

 //===--------------------------------------------------------------------===//
 // Transform Function
 //===--------------------------------------------------------------------===//
  static node::G_Node_Ptr Transform(node::Node_Ptr node_ptr,
                                    ui number_of_nodes);

 private:
  Node_Transformer() {};
};

} // End of transformer namespace
} // End of ursus namespace
