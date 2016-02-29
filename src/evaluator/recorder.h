#pragma once

#include "common/types.h"

namespace ursus {
namespace evaluator {

class Recorder{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  Recorder(const Recorder &) = delete;
  Recorder &operator=(const Recorder &) = delete;
  Recorder(Recorder &&) = delete;
  Recorder &operator=(Recorder &&) = delete;

  // global singleton
  static Recorder& GetInstance(void);

 //===--------------------------------------------------------------------===//
 // Time Record
 //===--------------------------------------------------------------------===//
  void TimeRecordStart();
  float TimeRecordEnd();

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
  Recorder() {}

  cudaEvent_t start_event, stop_event;
  float elapsed_time = 0.f;

  ui hit;

  ui count;
  ui root_count;

  ui pop_count;
  ui push_count;

//    float t_parent[7];
//    float t_skipLoggerer[7] ;
//    bool METHOD[7];
};

} // End of evaluator namespace
} // End of ursus namespace

