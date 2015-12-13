#include "common/evaluator.h"

int main(int argc, char** argv){

  auto& evaluator = ursus::Evaluator::GetInstance();
  if( !evaluator.ParseArgs(argc, argv))  {
    evaluator.PrintHelp(argv);
    return -1;
  }

  // TODO : Build Index

  // TODO : Search
  return 0;
}
