#pragma once

#include "common/macro.h"

namespace ursus {

class Logger{
 public:
 //===--------------------------------------------------------------------===//
 // Consteructor/Destructor
 //===--------------------------------------------------------------------===//
  Logger()  {
    // TODO :: I don't know what this one does ..
    for( unsigned range(i, 0, 7))  {
      t_time[i] = 0.0f;
      t_visit[i] = .0f;
      t_rootVisit[i] = .0f;
      t_pop[i] = .0f;
      t_push[i] = .0f;
      t_parent[i] = .0f;
      t_skipLoggerer[i] = .0f;
    }
  };
  ~Logger(){
  }



 //===--------------------------------------------------------------------===//
 // Accessors
 //===--------------------------------------------------------------------===//
//  unsigned int GetDims(void) const;
//  unsigned int GetBits(void) const;
//
//  void SetDims(unsigned int number_of_dimensions);
//  void SetBits(unsigned int number_of_bits);
//
//  // Get a string representation for debugging
//  friend std::ostream &operator<<(std::ostream &os, const Logger &logger);

 //===--------------------------------------------------------------------===//
 // Members
 //===--------------------------------------------------------------------===//
 private:
    float t_time[7];
    float t_visit[7];
    float t_rootVisit[7];
    float t_pop[7];
    float t_push[7];
    float t_parent[7];
    float t_skipLoggerer[7] ;
    bool METHOD[7];
 
};

} // End of ursus namespace

