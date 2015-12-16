#include "common/macro.h"
#include "node/branch.h"
#include "tree/mphr.h"

#include <stdio.h>
#include <iostream>

namespace ursus {
namespace tree {

MPHR::MPHR(){
  tree_type = TREE_TYPE_MPHR;
}

bool MPHR::Build(std::shared_ptr<io::DataSet> input_data_set){
  std::cout << "Build MPHR Tree" << std::endl;

  auto number_of_data = input_data_set->GetNumberOfData();
  number_of_dimensions = input_data_set->GetNumberOfDims();
  auto points = input_data_set->GetPoints();

  // create branches
  std::vector<node::Branch> branches(number_of_data);

  // create rect
  std::vector<Point> rect_points(number_of_dimensions*2);

  for( int range(i, 0, number_of_data)) {
    for( int range(j, 0, number_of_dimensions)) {
      rect_points[j] = rect_points[j+number_of_dimensions] = points[i*number_of_dimensions+j];
    }
    branches[i].SetRect(rect_points);
  }

  // assign indexes to branches 

  //For debugging
  for( int range(i, 0, 10)) {
    std::cout << branches[i] << std::endl;
  }

  return true;
}

int MPHR::Search(std::shared_ptr<io::DataSet> query_data_set){
return -1;
}


//void MPHR::AssignHilbertIndexes(
//  Branch b;
//  for(int i=0;i<NUMDATA;i++){
//    bitmask_t coord[NUMDIMS];
//    for(int j=0;j<NUMDIMS;j++){
//      coord[j] = (bitmask_t) (1000000*b.rect.boundary[j]);
//    }
//
//    //synthetic
//    if( !strcmp(DATATYPE, "high")){
//
//      if( NUMDIMS == 2 )
//        b.hIndex = hilbert_c2i(2, 31, coord);
//      else
//        b.hIndex = hilbert_c2i(3, 21, coord);
//    }
//    else{
//      b.hIndex = hilbert_c2i(3, 20, coord);
//    }
//
//    keys[i] = b.hIndex;
//    data[i] = b;
//  }

} // End of tree namespace
} // End of ursus namespace

