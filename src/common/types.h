#pragma once

#include <string>

namespace ursus {

typedef float Point;
typedef Point* Point_Ptr;
typedef unsigned int ui;
typedef unsigned long ul;
typedef unsigned long long ull;

//===--------------------------------------------------------------------===//
// DataSet
//===--------------------------------------------------------------------===//
enum DataSetType  {
  DATASET_TYPE_INVALID = -1,
  DATASET_TYPE_BINARY = 1
};

//===--------------------------------------------------------------------===//
// DataType
//===--------------------------------------------------------------------===//
enum DataType  {
  DATA_TYPE_INVALID = -1,
  DATA_TYPE_REAL = 1,
  DATA_TYPE_SYNTHETIC = 2
};


//===--------------------------------------------------------------------===//
// Node
//===--------------------------------------------------------------------===//
enum NodeType  {
  NODE_TYPE_INVALID = -1,
  NODE_TYPE_ROOT = 1,
  NODE_TYPE_INTERNAL = 2,
  NODE_TYPE_LEAF = 3
};

//===--------------------------------------------------------------------===//
// Tree
//===--------------------------------------------------------------------===//
enum TreeType  {
  TREE_TYPE_INVALID = -1,
  TREE_TYPE_MPHR = 1,
  TREE_TYPE_HYBRID =2
};

//TODO :: Rename bitmask_t to another one
/* define the bitmask_t type as an integer of sufficient size */
typedef unsigned long long bitmask_t;
/* define the halfmask_t type as an integer of 1/2 the size of bitmask_t */
typedef unsigned long halfmask_t;

//===--------------------------------------------------------------------===//
// Hilbert Curve
//===--------------------------------------------------------------------===//
/*
 * Readers and writers of bits
 */

typedef bitmask_t (*BitReader) (unsigned nDims, unsigned nBytes, char const* c, unsigned y);
typedef void (*BitWriter) (unsigned d, unsigned nBytes, char* c, unsigned y, int fold);

//===--------------------------------------------------------------------===//
// Transformers
//===--------------------------------------------------------------------===//

std::string DataSetTypeToString(DataSetType type);
DataSetType StringToDataSetType(std::string str);

std::string DataTypeToString(DataType type);
DataType StringToDataType(std::string str);

std::string NodeTypeToString(NodeType type);
NodeType StringToNodeType(std::string str);

std::string TreeTypeToString(TreeType type);
TreeType StringToTreeType(std::string str);

} // End of ursus namespace
