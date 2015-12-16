#include "common/macro.h"
#include "io/dataset.h"

#include <cassert>

namespace ursus {
namespace io {

DataSet::DataSet(unsigned int number_of_dimensions, unsigned int number_of_data,
                 std::string data_set_path, DataSetType data_set_type)
  : number_of_dimensions(number_of_dimensions), number_of_data(number_of_data),
    data_set_path(data_set_path), data_set_type(data_set_type) {

  // read data from data_set_path
  std::ifstream infile; 

  switch(data_set_type) {
    case DATASET_TYPE_BINARY:
      infile.open(data_set_path, std::ios::in | std::ios::binary);
      break;

    default:
      infile.open(data_set_path, std::ifstream::in);
  }

  points.resize(number_of_dimensions*number_of_data);

  infile.read(reinterpret_cast<char*>(&points[0]), 
              sizeof(Point)*number_of_data*number_of_dimensions);

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

std::vector<Point> DataSet::GetPoints(void) const{ 
  return points; 
}

// Get a string representation
std::ostream &operator<<(std::ostream &os, const DataSet &dataset) {
  os << " DataSet : " << std::endl
     << " Number of dimensions = " << dataset.GetNumberOfDims() << std::endl
     << " Number of data = " << dataset.GetNumberOfData() << std::endl
     << " DataSet path = " << dataset.GetDataSetPath() << std::endl
     << " DataSet type = " << DataSetTypeToString(dataset.GetDataSetType()) << std::endl;

  return os;
}

} // End of io namespace
} // End of ursus namespace
