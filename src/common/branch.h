#pragma once

#include "common/rect.h"
#include "common/node.h"

namespace ursus {

class Branch {
  private:
    Rect rect;
    unsigned long long index;
    Node *child; 
};

} // End of ursus namespace
