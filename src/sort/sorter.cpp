#include "sort/sorter.h"

#include "sort/thrust_sorter.h"
#include "sort/parallel_sorter.h"

#include "common/logger.h"

namespace ursus {
namespace sort {

bool Sorter::Sort(std::vector<node::Branch> &branches) {

  bool ret;

  // calculate size(MB) of branch
  auto size_for_branch = branches.size()*sizeof(node::Branch);

  // get the used and total size on GPU
  size_t avail, total;
  cudaMemGetInfo( &avail, &total );
  size_t used = total-avail;

  // if device memory size is not enough, sort on CPU
  if( (used+size_for_branch)/(double)total > 0.5) {
    ret = Parallel_Sorter::Sort(branches);
  } else { 
    ret = Thrust_Sorter::Sort(branches);
  }

  return ret;
}

} // End of sort namespace
} // End of ursus namespace



