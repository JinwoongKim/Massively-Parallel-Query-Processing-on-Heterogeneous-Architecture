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
          DataSetType data_set_type) 
  : number_of_dimensions(number_of_dimensions), number_of_data(number_of_data),
    data_set_path(data_set_path), data_set_type(data_set_type) {

    points = new Point[number_of_dimensions*number_of_data];

    // read data from data_set_path
    std::ifstream infile; 

    switch(data_set_type) {
      case DATASET_TYPE_BINARY:
        infile.open(data_set_path, std::ifstream::binary);
        break;

      default:
        infile.open(data_set_path, std::ifstream::in);
    }
    infile.read(reinterpret_cast<char*>(&points[0]), 
                sizeof(Point)*number_of_data*number_of_dimensions);


    //For debugging
    for( int range(i, 0, 5)) {
      std::cout << i << std::endl;
    }
  };

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
  unsigned int number_of_dimensions = 0;

  // # of data to be indexed
  unsigned int number_of_data = 0;

  // DataSet path
  std::string data_set_path;

  // data type
  DataSetType data_set_type = DATASET_TYPE_INVALID;

  // data points
  Point *points;
};

} // End of io namespace
} // End of ursus namespace

