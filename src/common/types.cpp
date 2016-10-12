#include "common/types.h"

namespace ursus {

//===--------------------------------------------------------------------===//
// DataSetType <--> String Utilities
//===--------------------------------------------------------------------===//

std::string DataSetTypeToString(DataSetType type) {
  std::string ret;

  switch (type) {
    case (DATASET_TYPE_INVALID):
      return "DATASET_TYPE_INVALID";
    case (DATASET_TYPE_BINARY):
      return "DATASET_TYPE_BINARY";
    default: {
      char buffer[32];
      ::snprintf(buffer, 32, "UNKNOWN[%d] ", type);
      ret = buffer;
    }
  }
  return (ret);
}

DataSetType StringToDataSetType(std::string str) {
  if (str == "DATASET_TYPE_INVALID") {
    return DATASET_TYPE_INVALID;
  } else if (str == "DATASET_TYPE_BINARY") {
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
      return "DATA_TYPE_INVALID";
    case (DATA_TYPE_REAL):
      return "DATA_TYPE_REAL";
    case (DATA_TYPE_SYNTHETIC):
      return "DATA_TYPE_SYNTHETIC";
    default: {
      char buffer[32];
      ::snprintf(buffer, 32, "UNKNOWN[%d] ", type);
      ret = buffer;
    }
  }
  return (ret);
}

DataType StringToDataType(std::string str) {
  if (str == "DATA_TYPE_INVALID") {
    return DATA_TYPE_INVALID;
  } else if (str == "DATA_TYPE_REAL") {
    return DATA_TYPE_REAL;
  } else if (str == "DATA_TYPE_SYNTHETIC") {
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
      return "NODE_TYPE_INVALID";
    case (NODE_TYPE_INTERNAL):
      return "NODE_TYPE_INTERNAL";
    case (NODE_TYPE_EXTENDLEAF):
      return "NODE_TYPE_EXTENDLEAF";
    case (NODE_TYPE_LEAF):
      return "NODE_TYPE_LEAF";
    default: {
      char buffer[32];
      ::snprintf(buffer, 32, "UNKNOWN[%d] ", type);
      ret = buffer;
    }
  }
  return (ret);
}

NodeType StringToNodeType(std::string str) {
  if (str == "NODE_TYPE_INVALID") {
    return NODE_TYPE_INVALID;
  } else if (str == "NODE_TYPE_INTERNAL") {
    return NODE_TYPE_INTERNAL;
  } else if (str == "NODE_TYPE_EXTENDLEAF") {
    return NODE_TYPE_EXTENDLEAF;
  } else if (str == "NODE_TYPE_LEAF") {
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
      return "TREE_TYPE_INVALID";
    case (TREE_TYPE_MPHR):
      return "TREE_TYPE_MPHR";
    case (TREE_TYPE_MPHR_PARTITION):
      return "TREE_TYPE_MPHR_PARTITION";
    case (TREE_TYPE_HYBRID):
      return "TREE_TYPE_HYBRID";
    case (TREE_TYPE_BVH):
      return "TREE_TYPE_BVH";
    case (TREE_TYPE_RTREE):
      return "TREE_TYPE_RTREE";
    default: {
      char buffer[32];
      ::snprintf(buffer, 32, "UNKNOWN[%d] ", type);
      ret = buffer;
    }
  }
  return (ret);
}

TreeType StringToTreeType(std::string str) {
  if (str == "TREE_TYPE_INVALID") {
    return TREE_TYPE_INVALID;
  } else if (str == "TREE_TYPE_MPHR") {
    return TREE_TYPE_MPHR;
  } else if (str == "TREE_TYPE_MPHR_PARTITION") {
    return TREE_TYPE_MPHR_PARTITION;
  } else if (str == "TREE_TYPE_HYBRID") {
    return TREE_TYPE_HYBRID;
  } else if (str == "TREE_TYPE_BVH") {
    return TREE_TYPE_BVH;
  } else if (str == "TREE_TYPE_RTREE") {
    return TREE_TYPE_RTREE;
  }
  return TREE_TYPE_INVALID;
}

//===--------------------------------------------------------------------===//
// ClusterType <--> String Utilities
//===--------------------------------------------------------------------===//

std::string ClusterTypeToString(ClusterType type) {
  std::string ret;

  switch (type) {
    case (CLUSTER_TYPE_INVALID):
      return "CLUSTER_TYPE_INVALID";
    case (CLUSTER_TYPE_NONE):
      return "CLUSTER_TYPE_NONE";
    case (CLUSTER_TYPE_HILBERT):
      return "CLUSTER_TYPE_HILBERT";
    case (CLUSTER_TYPE_KMEANSHILBERT):
      return "CLUSTER_TYPE_KMEANSHILBERT";
    default: {
      char buffer[32];
      ::snprintf(buffer, 32, "UNKNOWN[%d] ", type);
      ret = buffer;
    }
  }
  return (ret);
}

ClusterType StringToClusterType(std::string str) {
  if (str == "CLUSTER_TYPE_INVALID") {
    return CLUSTER_TYPE_INVALID;
  } else if (str == "CLUSTER_TYPE_NONE") {
    return CLUSTER_TYPE_NONE;
  } else if (str == "CLUSTER_TYPE_HILBERT") {
    return CLUSTER_TYPE_HILBERT;
  } else if (str == "CLUSTER_TYPE_KMEANSHILBERT") {
    return CLUSTER_TYPE_KMEANSHILBERT;
  }
  return CLUSTER_TYPE_INVALID;
}

} // End of ursus namespace

