#include "tree/tree.h"

#include "common/macro.h"
#include "mapper/hilbert_mapper.h"

namespace ursus {
namespace tree {

std::vector<node::Branch> Tree::CreateBranches(std::shared_ptr<io::DataSet> input_data_set) {
  number_of_dimensions = input_data_set->GetNumberOfDims();
  number_of_data = input_data_set->GetNumberOfData();
  auto points = input_data_set->GetPoints();

  // create branches
  std::vector<node::Branch> branches(number_of_data);

  for( int range(i, 0, number_of_data)) {
    branches[i].SetMBB(&points[i*number_of_dimensions]);
  }

  return branches;
}

bool Tree::AssignHilbertIndexToBranches(std::vector<node::Branch> &branches) {
  unsigned int number_of_bits = (number_of_dimensions>2) ? 20:31;

  for(int range(i, 0, branches.size())) {
    auto points = branches[i].GetPoints();
    auto hilbert_index = mapper::Hilbert_Mapper::MappingIntoSingle(number_of_dimensions,
                                                                   number_of_bits, points);
    branches[i].SetIndex(hilbert_index);
  }

  return true;
}

} // End of tree namespace
} // End of ursus namespace
