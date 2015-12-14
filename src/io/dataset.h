#pragma once

#include "common/macro.h"
#include "common/types.h"

#include <iostream>
#include <fstream>

namespace ursus {
namespace io {

class DataSet{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  DataSet(unsigned int number_of_dimensions,
          unsigned int number_of_data,
          std::string data_set_path,
          DataSetType data_set_type);

  ~DataSet(){
    delete[] points;
  }

 //===--------------------------------------------------------------------===//
 // Accessors
 //===--------------------------------------------------------------------===//
  unsigned int GetNumberOfDims(void) const;

  unsigned int GetNumberOfData(void) const;

  std::string GetDataSetPath(void) const;

  DataSetType GetDataSetType(void) const;

  // Get a string representation for debugging
  friend std::ostream &operator<<(std::ostream &os, const DataSet &dataset);

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  // # of dims
  unsigned int number_of_dimensions;

  // # of data to be indexed
  unsigned int number_of_data;

  // DataSet path
  std::string data_set_path;

  // data type
  DataSetType data_set_type = DATASET_TYPE_INVALID;

  // data points
  Point *points;
};

} // End of io namespace
} // End of ursus namespace

