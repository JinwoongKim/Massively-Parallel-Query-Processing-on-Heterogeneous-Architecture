#include "io/dataset.h"

#include "common/macro.h"

#include <cassert>

namespace ursus {
namespace io {

DataSet::DataSet(unsigned int number_of_dimensions, unsigned int number_of_data,
                 std::string data_set_path, DataSetType data_set_type, DataType data_type,
                 ClusterType cluster_type)
  : number_of_dimensions(number_of_dimensions), number_of_data(number_of_data),
    data_set_path(data_set_path), data_set_type(data_set_type), 
    data_type(data_type), cluster_type(cluster_type){

  // read data from data_set_path
  std::ifstream input_stream; 

  switch(data_set_type) {
    case DATASET_TYPE_BINARY:
      input_stream.open(data_set_path, std::ios::in | std::ios::binary);
      break;

    default:
      input_stream.open(data_set_path, std::ifstream::in);
  }

  // print out an error message when it was failed to be opened
  if(!input_stream){
    std::cerr << "Failed to open a file(" << data_set_path << ")\n";
    exit(1);
  } 
 

  points.resize(number_of_dimensions*number_of_data);

  input_stream.read(reinterpret_cast<char*>(&points[0]), 
              sizeof(Point)*number_of_data*number_of_dimensions);

  std::cout << *this << std::endl;
  input_stream.close();
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

DataType DataSet::GetDataType(void) const{ 
  return data_type; 
}

ClusterType DataSet::GetClusterType(void) const{ 
  return cluster_type; 
}

std::vector<Point> DataSet::GetPoints(void) const{ 
  return points; 
}

Point* DataSet::GetDeviceQuery(ui number_of_search) const{ 
  Point* d_query;
  cudaErrCheck(cudaMalloc((void**) &d_query, sizeof(Point)*GetNumberOfDims()*2*number_of_search));
  cudaErrCheck(cudaMemcpy(d_query, &points[0], sizeof(Point)*GetNumberOfDims()*2*number_of_search,
               cudaMemcpyHostToDevice));
  return d_query; 
}

// Get a string representation
std::ostream &operator<<(std::ostream &os, const DataSet &dataset) {
  os << " DataSet : " << std::endl
     << " Number of dimensions = " << dataset.GetNumberOfDims() << std::endl
     << " Number of data = " << dataset.GetNumberOfData() << std::endl
     << " DataSet path = " << dataset.GetDataSetPath() << std::endl
     << " DataSet type = " << DataSetTypeToString(dataset.GetDataSetType()) << std::endl
     << " Data type = " << DataTypeToString(dataset.GetDataType()) << std::endl
     << " Cluster type = " << ClusterTypeToString(dataset.GetClusterType()) << std::endl;

  return os;
}

} // End of io namespace
} // End of ursus namespace
