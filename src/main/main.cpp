#include "evaluator/evaluator.h"

int main(int argc, char** argv){

  // Initialize evaluator which will build the indexing structure and measure
  // the search performance
  auto& evaluator = ursus::evaluator::Evaluator::GetInstance();

  //TODO :: Setting dataset and indexing structure inside evaluator but not now XD
  if( !evaluator.Initialize(argc, argv))  {
    return -1;
  }

  // TODO : Build Index
  evaluator.Build();

  // TODO : Search
  return 0;
}
