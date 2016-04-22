#include "sort/sorter.h"

#include "common/logger.h"
#include "evaluator/evaluator.h"
#include "sort/thrust_sorter.h"
#include "sort/parallel_sorter.h"

namespace ursus {
namespace sort {

bool Sorter::Sort(std::vector<node::Branch> &branches) {
  bool ret;

  // calculate size(MB) of branch
  auto size_for_branch = branches.size()*sizeof(node::Branch);

  // get the used and total size on GPU
  size_t used = evaluator::Evaluator::GetUsedMem();
  size_t total = evaluator::Evaluator::GetTotalMem();

  // if device doesn't have enough space, sort the data on CPU
  if( (used+size_for_branch)/(double)total > 0.5) {
    ret = Parallel_Sorter::Sort(branches);
  } else { 
    ret = Thrust_Sorter::Sort(branches);
  }

  return ret;
}

} // End of sort namespace
} // End of ursus namespace



