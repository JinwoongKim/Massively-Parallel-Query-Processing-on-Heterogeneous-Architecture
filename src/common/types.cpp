#include "common/types.h"

namespace ursus {

//===--------------------------------------------------------------------===//
// DataSetType <--> String Utilities
//===--------------------------------------------------------------------===//

std::string DataSetTypeToString(DataSetType type) {
  std::string ret;

  switch (type) {
    case (DATASET_TYPE_INVALID):
      return "INVALID";
    case (DATASET_TYPE_BINARY):
      return "BINARY";
    default: {
      char buffer[32];
      ::snprintf(buffer, 32, "UNKNOWN[%d] ", type);
      ret = buffer;
    }
  }
  return (ret);
}

DataSetType StringToDataSetType(std::string str) {
  if (str == "INVALID") {
    return DATASET_TYPE_INVALID;
  } else if (str == "BINARY") {
    return DATASET_TYPE_BINARY;
  }
  return DATASET_TYPE_INVALID;
}

//===--------------------------------------------------------------------===//
// DataType <--> String Utilities
//===--------------------------------------------------------------------===//

std::string DataTypeToString(DataType type) {
  std::string ret;

  switch (type) {
    case (DATA_TYPE_INVALID):
      return "INVALID";
    case (DATA_TYPE_REAL):
      return "BINARY";
    case (DATA_TYPE_SYNTHETIC):
      return "SYNTHETIC";
    default: {
      char buffer[32];
      ::snprintf(buffer, 32, "UNKNOWN[%d] ", type);
      ret = buffer;
    }
  }
  return (ret);
}

DataType StringToDataType(std::string str) {
  if (str == "INVALID") {
    return DATA_TYPE_INVALID;
  } else if (str == "REAL") {
    return DATA_TYPE_REAL;
  } else if (str == "SYNTHETIC") {
    return DATA_TYPE_SYNTHETIC;
  }
  return DATA_TYPE_INVALID;
}

//===--------------------------------------------------------------------===//
// NodeType <--> String Utilities
//===--------------------------------------------------------------------===//

std::string NodeTypeToString(NodeType type) {
  std::string ret;

  switch (type) {
    case (NODE_TYPE_INVALID):
      return "INVALID";
    case (NODE_TYPE_INTERNAL):
      return "INTERNAL";
    case (NODE_TYPE_LEAF):
      return "LEAF";
    default: {
      char buffer[32];
      ::snprintf(buffer, 32, "UNKNOWN[%d] ", type);
      ret = buffer;
    }
  }
  return (ret);
}

NodeType StringToNodeType(std::string str) {
  if (str == "INVALID") {
    return NODE_TYPE_INVALID;
  } else if (str == "INTERNAL") {
    return NODE_TYPE_INTERNAL;
  } else if (str == "LEAF") {
    return NODE_TYPE_LEAF;
  }
  return NODE_TYPE_INVALID;
}

//===--------------------------------------------------------------------===//
// TreeType <--> String Utilities
//===--------------------------------------------------------------------===//

std::string TreeTypeToString(TreeType type) {
  std::string ret;

  switch (type) {
    case (TREE_TYPE_INVALID):
      return "INVALID";
    case (TREE_TYPE_MPHR):
      return "MPHR";
    case (TREE_TYPE_HYBRID):
      return "HYBRID";
    default: {
      char buffer[32];
      ::snprintf(buffer, 32, "UNKNOWN[%d] ", type);
      ret = buffer;
    }
  }
  return (ret);
}

TreeType StringToTreeType(std::string str) {
  if (str == "INVALID") {
    return TREE_TYPE_INVALID;
  } else if (str == "MPHR") {
    return TREE_TYPE_MPHR;
  } else if (str == "HYBRID") {
    return TREE_TYPE_HYBRID;
  }
  return TREE_TYPE_INVALID;
}

} // End of ursus namespace

