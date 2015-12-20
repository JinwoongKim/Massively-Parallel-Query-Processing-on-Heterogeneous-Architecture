#pragma once

#include "common/global.h"
#include "common/types.h"

namespace ursus {
namespace node {

typedef class G_Node* G_Node_Ptr;

class G_Node{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  // transformed branches
  Point point[GetNumberOfDims()*2*GetNumberOfDegrees()];
  ull index[GetNumberOfDegrees()];
  G_Node_Ptr child[GetNumberOfDegrees()];

  // node type
  NodeType node_type = NODE_TYPE_INVALID;

  // distance from root
  ui level;
};

} // End of node namespace
} // End of ursus namespace
