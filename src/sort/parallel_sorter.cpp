#include "sort/parallel_sorter.h"

#include "common/logger.h"
#include "evaluator/recorder.h"

#include <algorithm>  
#include <thread>
#include <functional>
#include "tbb/parallel_sort.h"

namespace ursus {
namespace sort {

void Thread_Assign(std::vector<node::Branch> &branches, 
                   ui tid, ui number_of_threads) {

  for(ui offset = tid; offset < branches.size() ; offset+=number_of_threads) {
    branches[offset].SetIndex(offset+1);
  }
}

bool Parallel_Sorter::Sort(std::vector<node::Branch> &branches) {

  auto& recorder = evaluator::Recorder::GetInstance();
  recorder.TimeRecordStart();

  tbb::parallel_sort(branches.begin(), branches.end());

  const size_t number_of_threads = std::thread::hardware_concurrency();

  // parallel for loop using c++ std 11 
  {
    std::vector<std::thread> threads;

    //Launch a group of threads
    for (ui thread_itr = 0; thread_itr<number_of_threads; thread_itr++){
      threads.push_back(std::thread(Thread_Assign, std::ref(branches), thread_itr, number_of_threads));
    }

    //Join the threads with the main thread
    for(auto &thread : threads){
      thread.join();
    }
  }

  // print out sorting time on the GPU
  auto elapsed_time = recorder.TimeRecordEnd();
  LOG_INFO("Sort Time on CPU (%zu threads) = %.6fs", number_of_threads, elapsed_time/1000.0f);

  return true;
}

} // End of sort namespace
} // End of ursus namespace

