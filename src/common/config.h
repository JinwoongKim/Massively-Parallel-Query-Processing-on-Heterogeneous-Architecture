#pragma once

namespace ursus {
  __host__ __device__ constexpr unsigned int GetNumberOfDims() { return 3; }

  __host__ __device__ constexpr unsigned int GetNumberOfDegrees() { return 3; }

  __host__ __device__ constexpr unsigned int GetNumberOfBlocks() { return 128; }

  __host__ __device__ constexpr unsigned int GetNumberOfThreads() { return 64; }

} // End of ursus namespace
