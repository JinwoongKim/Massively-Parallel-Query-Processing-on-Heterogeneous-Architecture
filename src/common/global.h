#pragma once

namespace ursus {
  __host__ __device__ constexpr unsigned int GetNumberOfDims() { return 3; }

  __host__ __device__ constexpr unsigned int GetNumberOfDegrees() { return 128; }
} // End of ursus namespace

// Another way
// file1.hpp <-- This is now a HEADER not a CPP
//enum constant { size = 10 };
//
//// mainfile.cpp
//#include "file1.hpp"
//
//void main()
//{
//  int mas[constant::size];
//}
