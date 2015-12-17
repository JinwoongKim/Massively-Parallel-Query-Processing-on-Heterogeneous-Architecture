#pragma once

#include "node/branch.h"

#include <vector>

namespace ursus {
namespace sort {

class Thrust_Sort{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  Thrust_Sort();

 //===--------------------------------------------------------------------===//
 // Main Function
 //===--------------------------------------------------------------------===//

  /**
   * Sort the data 
   */
  static bool Sort(std::vector<node::Branch> &branches);
};

} // End of sort namespace
} // End of ursus namespace
