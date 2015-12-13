#pragma once

#include "common/logger.h"

#include <iostream>
#include <string>

namespace ursus {

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
  // Guide & Parse
  //===--------------------------------------------------------------------===//
 
  // Print out usage to users
  void PrintHelp(char **argv) const;
 
  bool ParseArgs(int argc, char **argv);
 
  // Get a string representation for debugging
  friend std::ostream &operator<<(std::ostream &os, const Evaluator &evaluator);

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  Evaluator(){};

  // # of data to be indexed
  unsigned number_of_data = 0;

  // # of searches
  unsigned number_of_searches = 0;

  // # of partitioned trees
  // 1 means braided version(no partition), 
  // 2 or more indicate partitioned version
  unsigned number_of_partitioned_trees = 1;

  // # of thread_blocks
  unsigned number_of_thread_blocks = 128;

  // # of cpu cores to be used for evaluation
  unsigned number_of_cpu_cores = 0;
  
  //TODO Why is this string instead of float??
  std::string selectivity="0.01";

  std::string query_size;
  
  // # of dims
  unsigned number_of_dimensions;
  
  // # of bits which represent each dimension 
  // TODO Do we need this ??
  unsigned number_of_bits;

  // Measure and record time and count  
  Logger logger;

  char** ch_root;
  char** cd_root;
 
/*
  //FIXME :: do later,
  BUILD_TYPE = 0;
  // workload offset
  unsigned workload_offset = 0;
  // workload offset
  unsigned policy = 0;
  DEVICE_ID = 0;
  keepDoing = 0;
*/
 
};

} // End of ursus namespace

