#pragma once

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

//    // TODO :: I don't know what this one does ..
//    for( unsigned range(i, 0, 7))  {
//      t_time[i] = 0.0f;
//      t_visit[i] = .0f;
//      t_rootVisit[i] = .0f;
//      t_pop[i] = .0f;
//      t_push[i] = .0f;
//      t_parent[i] = .0f;
//      t_skipLoggerer[i] = .0f;
//    }
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

//    float t_time[7];
//    float t_visit[7];
//    float t_rootVisit[7];
//    float t_pop[7];
//    float t_push[7];
//    float t_parent[7];
//    float t_skipLoggerer[7] ;
//    bool METHOD[7];
 
};

} // End of evaluator namespace
} // End of ursus namespace

