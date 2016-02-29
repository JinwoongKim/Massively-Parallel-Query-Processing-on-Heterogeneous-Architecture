#pragma once

#include "io/dataset.h"
#include "tree/tree.h"

#include <iostream>
#include <string>
#include <memory>

namespace ursus {
namespace evaluator {

class Evaluator{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  Evaluator(const Evaluator &) = delete;
  Evaluator &operator=(const Evaluator &) = delete;
  Evaluator(Evaluator &&) = delete;
  Evaluator &operator=(Evaluator &&) = delete;

  // global singleton
  static Evaluator& GetInstance(void);

  //===--------------------------------------------------------------------===//
  // Initialize
  //===--------------------------------------------------------------------===//
  bool Initialize(int argc, char **argv);

  bool ReadDataSet(void);

  bool ReadQuerySet(void);
 
  int SetDevice(ui number_of_gpus);

  bool Build(void);

  bool Search(void);

  // Print out usage to users
  void PrintHelp(char **argv) const;
 
  bool ParseArgs(int argc, char **argv);

  // Get a string representation for debugging
  friend std::ostream &operator<<(std::ostream &os, const Evaluator &evaluator);

 private:
  Evaluator() {};

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
  // # of data to be indexed
  ui number_of_data = 0;

  // # of searches
  ui number_of_searches = 0;

  // # of partitioned trees
  // 1 means braided version(no partition), 
  // 2 or more indicate partitioned version
  ui number_of_partitioned_trees = 1;

  // # of cpu cores to be used for evaluation
  ui number_of_cpu_cores = 0;
  
  std::string selectivity="0.01";
  std::string query_size;

  // # of gpus
  ui number_of_gpus = 1;
  
  // Measure and record time and count  
  //Logger logger;
  
  std::shared_ptr<io::DataSet> input_data_set;
  std::shared_ptr<io::DataSet> query_data_set;

  std::vector<std::unique_ptr<tree::Tree>> trees;
};

} // End of evaluator namespace
} // End of ursus namespace

