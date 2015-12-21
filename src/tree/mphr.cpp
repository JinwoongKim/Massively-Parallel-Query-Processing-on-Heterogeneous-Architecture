#include "tree/mphr.h"

#include "sort/thrust_sort.h"

#include "common/macro.h"
#include "common/logger.h"

#include <cassert>

namespace ursus {
namespace tree {

MPHR::MPHR() {
  tree_type = TREE_TYPE_MPHR;
}

bool MPHR::Build(std::shared_ptr<io::DataSet> input_data_set){
  LOG_INFO("Build MPHR Tree");
  bool ret = false;

 //===--------------------------------------------------------------------===//
 // Create branches
 //===--------------------------------------------------------------------===//
  std::vector<node::Branch> branches = CreateBranches(input_data_set);

 //===--------------------------------------------------------------------===//
 // Assign Hilbert Id to branches
 //===--------------------------------------------------------------------===//
  // TODO  have to choose policy later
  ret = AssignHilbertIndexToBranches(branches);
  assert(ret);

 //===--------------------------------------------------------------------===//
 // Sort the branches on the GPU
 //===--------------------------------------------------------------------===//
  ret = sort::Thrust_Sort::Sort(branches);
  assert(ret);

 //===--------------------------------------------------------------------===//
 // Transform the branches into SOA fashion 
 //===--------------------------------------------------------------------===//

 //===--------------------------------------------------------------------===//
 // Build the tree in a bottop-up fashion on the GPU
 //===--------------------------------------------------------------------===//
  // Bottom-up building on the GPU

  // Transfer to the GPU

  //For debugging
  for( int range(i, 0, 10)) {
    std::cout << branches[i] << std::endl;
  }

  return true;
}


int MPHR::Search(std::shared_ptr<io::DataSet> query_data_set){
  return -1  ;
}


} // End of tree namespace
} // End of ursus namespace

