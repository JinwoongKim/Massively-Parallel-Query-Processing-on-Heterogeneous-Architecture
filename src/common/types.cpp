#include "common/types.h"

namespace ursus {

//===--------------------------------------------------------------------===//
// DataSetType <--> String Utilities
//===--------------------------------------------------------------------===//

std::string DataSetTypeToString(DataSetType type) {
  std::string ret;

  switch (type) {
    case (DATASET_TYPE_INVALID):
      return "TYPE_INVALID";
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
  if (str == "TYPE_INVALID") {
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
      return "TYPE_INVALID";
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
  if (str == "TYPE_INVALID") {
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
      return "TYPE_INVALID";
    case (NODE_TYPE_INTERNAL):
      return "INTERNAL";
    case (NODE_TYPE_EXTENDLEAF):
      return "EXTENDLEAF";
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
  if (str == "TYPE_INVALID") {
    return NODE_TYPE_INVALID;
  } else if (str == "INTERNAL") {
    return NODE_TYPE_INTERNAL;
  } else if (str == "EXTENDLEAF") {
    return NODE_TYPE_EXTENDLEAF;
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
      return "TYPE_INVALID";
    case (TREE_TYPE_MPHR):
      return "MPHR";
    case (TREE_TYPE_MPHR_PARTITIONED):
      return "MPHR_PARTITIONED
    case (TREE_TYPE_HYBRID):
      return "HYBRID";
    case (TREE_TYPE_RTREE):
      return "RTREE";
    default: {
      char buffer[32];
      ::snprintf(buffer, 32, "UNKNOWN[%d] ", type);
      ret = buffer;
    }
  }
  return (ret);
}

TreeType StringToTreeType(std::string str) {
  if (str == "TYPE_INVALID") {
    return TREE_TYPE_INVALID;
  } else if (str == "MPHR") {
    return TREE_TYPE_MPHR;
  } else if (str == "MPHR_PARTITIONED") {
    return TREE_TYPE_MPHR_PARTITIONED;
  } else if (str == "HYBRID") {
    return TREE_TYPE_HYBRID;
  } else if (str == "RTREE") {
    return TREE_TYPE_RTREE;
  }
  return TREE_TYPE_INVALID;
}

//===--------------------------------------------------------------------===//
// ScanType <--> String Utilities
//===--------------------------------------------------------------------===//

std::string ScanTypeToString(ScanType type) {
  std::string ret;

  switch (type) {
    case (SCAN_TYPE_INVALID):
      return "TYPE_INVALID";
    case (SCAN_TYPE_LEAF):
      return "LEAF";
    case (SCAN_TYPE_EXTENDLEAF):
      return "EXTENDLEAF";
    default: {
      char buffer[32];
      ::snprintf(buffer, 32, "UNKNOWN[%d] ", type);
      ret = buffer;
    }
  }
  return (ret);
}

ScanType StringToScanType(std::string str) {
  if (str == "TYPE_INVALID") {
    return SCAN_TYPE_INVALID;
  } else if (str == "LEAF") {
    return SCAN_TYPE_LEAF;
  } else if (str == "EXTENDLEAF") {
    return SCAN_TYPE_EXTENDLEAF;
  }
 
  return SCAN_TYPE_INVALID;
}




} // End of ursus namespace

