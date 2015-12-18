#include "sort/thrust_sort.h"

#include "common/logger.h"
#include "evaluator/recorder.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace ursus {
namespace sort {

bool Thrust_Sort::Sort(std::vector<node::Branch> &branches) {
  auto& recorder = evaluator::Recorder::GetInstance();

  LOG_INFO("Sort the data on the GPU");

  recorder.TimeRecordStart();

  // copy host to device
  thrust::device_vector<node::Branch> d_branches = branches;

  // sort the data
  thrust::sort(d_branches.begin(), d_branches.end());

  // copy back to host
  thrust::copy(d_branches.begin(), d_branches.end(), branches.begin());

  // print out sorting time on the GPU
  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Sort Time on GPU = %.6fs", elapsed_time/1000.0f);

  return true;
}


} // End of sort namespace
} // End of ursus namespace

