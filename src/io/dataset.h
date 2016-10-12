#pragma once

#include "common/types.h"

#include <iostream>
#include <fstream>
#include <vector>

namespace ursus {
namespace io {

class DataSet{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  DataSet(ui number_of_dimensions,
          ui number_of_data,
          std::string data_set_path,
          DataSetType data_set_type,
          DataType data_type,
          ClusterType cluster_type,
          std::string force_rebuild);

  ~DataSet(){
  }

 //===--------------------------------------------------------------------===//
 // Accessors
 //===--------------------------------------------------------------------===//
  unsigned int GetNumberOfDims(void) const;

  unsigned int GetNumberOfData(void) const;

  std::string GetDataSetPath(void) const;

  DataSetType GetDataSetType(void) const;

  DataType GetDataType(void) const;

  ClusterType GetClusterType(void) const;

  std::vector<Point> GetPoints(void) const;

  Point* GetDeviceQuery(ui number_of_search) const;

  bool IsRebuild(void) const;

  // Get a string representation for debugging
  friend std::ostream &operator<<(std::ostream &os, const DataSet &dataset);

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  // # of dims
  unsigned int number_of_dimensions;

  // # of data or query
  unsigned int number_of_data;

  // DataSet path
  std::string data_set_path;

  // data type
  DataSetType data_set_type = DATASET_TYPE_INVALID;

  // DataSet path
  DataType data_type;

  // Cluster Type
  ClusterType cluster_type;

  std::vector<Point> points;

  // dumped file path
  std::string force_rebuild;
};

} // End of io namespace
} // End of ursus namespace

