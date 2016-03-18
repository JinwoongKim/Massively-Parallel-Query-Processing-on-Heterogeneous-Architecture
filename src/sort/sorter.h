#pragma once

#include "node/branch.h"

#include <vector>

namespace ursus {
namespace sort {

class Sorter {
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  Sorter();

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
