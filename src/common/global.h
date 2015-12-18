#pragma once

namespace ursus {
  constexpr unsigned int GetNumberOfDims() { return 3; }
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
