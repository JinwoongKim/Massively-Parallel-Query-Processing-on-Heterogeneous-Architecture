#pragma once

#include <string>

namespace ursus {

typedef float Point;

//===--------------------------------------------------------------------===//
// DataSet
//===--------------------------------------------------------------------===//
enum DataSetType  {
  DATASET_TYPE_INVALID = -1,
  DATASET_TYPE_BINARY = 1
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

/* define the bitmask_t type as an integer of sufficient size */
//TODO :: Rename bitmask_t to another one
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

} // End of ursus namespace
