#pragma once

#include "node/node.h"
#include "node/g_node.h"

namespace ursus {
namespace transformer {

class Transformer{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  Transformer(const Transformer &) = delete;
  Transformer &operator=(const Transformer &) = delete;
  Transformer(Transformer &&) = delete;
  Transformer &operator=(Transformer &&) = delete;

  // global singleton
  static Transformer& GetInstance(void);

 //===--------------------------------------------------------------------===//
 // Transform Function
 //===--------------------------------------------------------------------===//
  static node::G_Node* Transform(node::Node* node,
                                    ui number_of_nodes);

 private:
  Transformer() {};
};

} // End of transformer namespace
} // End of ursus namespace
