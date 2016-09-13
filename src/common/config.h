#pragma once

namespace ursus {
  __host__ __device__ constexpr unsigned int GetNumberOfDims() { return 3; }

  __host__ __device__ constexpr unsigned int GetNumberOfDegrees() { return 32; }

  __host__ __device__ constexpr unsigned int GetNumberOfDegrees2() { return 32; }

  __host__ __device__ constexpr unsigned int GetNumberOfThreads() { return 128; }

  __host__ __device__ constexpr unsigned int GetNumberOfMAXBlocks() { return 256; }
} // End of ursus namespace
