#include "io/dataset.h"
#include <cassert>

namespace ursus {
namespace io {

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
