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

//===--------------------------------------------------------------------===//
// Cuda Function 
//===--------------------------------------------------------------------===//
__global__ 
void ReassignHilbertIndexes(node::Branch* branches, int number_of_data );

} // End of sort namespace
} // End of ursus namespace
