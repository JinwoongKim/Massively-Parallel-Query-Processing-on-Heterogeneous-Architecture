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
 
  int SetDevice();

  bool Build(void);

  bool Search(void);

  // Print out usage to users
  void PrintHelp(char **argv) const;

  void PrintMemoryUsageOftheGPU();

  static size_t GetUsedMem(void);

  static size_t GetAvailMem(void);

  static size_t GetTotalMem(void);

  bool ParseArgs(int argc, char **argv);

  void AddTrees(std::string index_type);

  DataType GetDataType(void);

  ClusterType GetClusterType(void);

  std::string GetDataPath(const DataType data_type) const;
 
  std::string GetQueryPath(const DataType data_type) const;

  std::string ToLowerCase(std::string str);

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
  ui number_of_search = 0;

  // # of repeat in Search function
  ui number_of_repeat = 1;

  ui number_of_partition = 1;

  // evaluation mode, if it's on, run a search function multiple time with
  // various settings
  ui EvaluationMode = 0;

  // # of cpu cores to be used for evaluation
  ui number_of_cpu_core = 0;

  // # of CUDA blocks
  ui number_of_cuda_blocks = 128;

  // # of cpu threads to process query concurrently 
  ui number_of_cpu_threads = 1;

  // scan type
  ui  scan_level = 1;
  
  std::string selectivity="0.01";

  std::string query_size;

  std::string s_data_type = "synthetic";

  std::string s_cluster_type= "hilbert";

  std::string s_force_rebuild= "no";

  TreeType UPPER_TREE_TYPE=TREE_TYPE_RTREE;

  // To control chunk_size in Hybrid indexing 
  ui chunk_size = 128;

  std::shared_ptr<io::DataSet> input_data_set;

  std::shared_ptr<io::DataSet> query_data_set;

  std::vector<std::shared_ptr<tree::Tree>> trees;
};

} // End of evaluator namespace
} // End of ursus namespace

