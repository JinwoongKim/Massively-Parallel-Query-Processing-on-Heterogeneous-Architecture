#include "tree/mphr.h"

#include <iostream>

namespace ursus {
namespace tree {

MPHR::MPHR(){
  tree_type = TREE_TYPE_MPHR;
}

bool MPHR::Build(io::DataSet *input_data_set){
  std::cout << "Build MPHR Tree" << std::endl;
  return true;
}

void MPHR::Search(io::DataSet *query_data_set){
}

} // End of tree namespace
} // End of ursus namespace

