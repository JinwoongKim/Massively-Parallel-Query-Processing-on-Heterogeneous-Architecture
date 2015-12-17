#include "sort/thrust_sort.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace ursus {
namespace sort {

struct branch_comp
{
  __host__ __device__
  bool operator()(node::Branch &b1, node::Branch &b2) {
    return b1.GetIndex() < b2.GetIndex();
  }
};

bool Thrust_Sort::Sort(std::vector<node::Branch> &branches) {
  std::cout << "Sort the data on the GPU" << std::endl;
  auto number_of_data = branches.size();

  // copy host to device
  thrust::device_vector<node::Branch> d_branches(number_of_data);
  thrust::copy(branches.begin(), branches.end(), d_branches.begin());

  // sort the data
  thrust::sort(d_branches.begin(), d_branches.end(), branch_comp());

  // copy back to host
  thrust::copy(d_branches.begin(), d_branches.end(), branches.begin());

  return true;
}


} // End of sort namespace
} // End of ursus namespace

