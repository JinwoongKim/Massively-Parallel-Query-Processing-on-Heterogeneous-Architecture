#include "tree/mphr.h"

#include "common/macro.h"

#include <iostream>

namespace ursus {
namespace tree {

MPHR::MPHR() {
  tree_type = TREE_TYPE_MPHR;
}

bool MPHR::Build(std::shared_ptr<io::DataSet> input_data_set){
  std::cout << "Build MPHR Tree" << std::endl;

  // create branches with points in input_data_set
  std::vector<node::Branch> branches = CreateBranches(input_data_set);

  AssignHilbertIndexToBranches(branches);

  // TODO  choose policy later
  // assign hilbert indexes to branches 

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

