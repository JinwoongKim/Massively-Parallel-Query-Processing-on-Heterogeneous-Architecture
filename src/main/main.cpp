#include "evaluator/evaluator.h"

int main(int argc, char** argv){

  auto& evaluator = ursus::evaluator::Evaluator::GetInstance();

  // Initialize evaluator which will build the indexing structure and measure
  // the search performance
  //TODO :: Setting dataset and indexing structure in parse argus function in evaluator class but not now XD
  if( !evaluator.Initialize(argc, argv))  {
    return -1;
  }

  evaluator.Build();

  evaluator.PrintMemoryUsageOftheGPU();

  evaluator.Search();
  return 0;
}
