#include "io/dataset.h"

#include <cassert>
#include <stdio.h>

namespace ursus {
namespace io {

DataSet::DataSet(unsigned int number_of_dimensions, unsigned int number_of_data,
                 std::string data_set_path, DataSetType data_set_type)
  : number_of_dimensions(number_of_dimensions), number_of_data(number_of_data),
    data_set_path(data_set_path), data_set_type(data_set_type) {
  points = new Point[number_of_dimensions*number_of_data];

  // read data from data_set_path
  std::ifstream infile; 

  switch(data_set_type) {
    case DATASET_TYPE_BINARY:
      infile.open(data_set_path, std::ios::in | std::ios::binary);
      break;

    default:
      infile.open(data_set_path, std::ifstream::in);
  }

  infile.read(reinterpret_cast<char*>(&points[0]), 
              sizeof(Point)*number_of_data*number_of_dimensions);

//  //For debugging
//  for( int range(i, 0, 10)) {
//    printf(" point : %.6f\n" ,points[i]);
//  }

  std::cout << *this << std::endl;
}


unsigned int DataSet::GetNumberOfDims(void) const{ 
  return number_of_dimensions; 
}

unsigned int DataSet::GetNumberOfData(void) const{ 
  return number_of_data; 
}

std::string DataSet::GetDataSetPath(void) const{ 
  return data_set_path; 
}

DataSetType DataSet::GetDataSetType(void) const{ 
  return data_set_type; 
}

// Get a string representation
std::ostream &operator<<(std::ostream &os, const DataSet &dataset) {
  os << " number of dimensions = " << dataset.GetNumberOfDims() << std::endl
     << " number of data = " << dataset.GetNumberOfData() << std::endl
     << " data set path = " << dataset.GetDataSetPath() << std::endl
     << " data set type = " << dataset.GetDataSetType() << std::endl;

  return os;
}

} // End of io namespace
} // End of ursus namespace
