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

} // End of ursus namespace
