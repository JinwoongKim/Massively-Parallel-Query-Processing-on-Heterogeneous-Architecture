#pragma once

#include "node/branch.h"

#include <vector>

namespace ursus {
namespace sort {

class Parallel_Sorter{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  Parallel_Sorter();

 //===--------------------------------------------------------------------===//
 // Main Function
 //===--------------------------------------------------------------------===//

  /*
   * Sort the data 
   */
  static bool Sort(std::vector<node::Branch> &branches);
};

} // End of sort namespace
} // End of ursus namespace
