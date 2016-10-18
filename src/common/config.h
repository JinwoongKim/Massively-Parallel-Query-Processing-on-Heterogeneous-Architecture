#pragma once

namespace ursus {
  __host__ __device__ constexpr unsigned int GetNumberOfDims() { return 3; }

  __host__ __device__ constexpr unsigned int GetNumberOfLeafNodeDegrees() { return 192; }

  __host__ __device__ constexpr unsigned int GetNumberOfUpperTreeDegrees() { return 128; }

  __host__ __device__ constexpr unsigned int GetNumberOfThreads() { return 192; }

  // For parallel reduction
  // TODO Rename it 
  __host__ __device__ constexpr unsigned int GetNumberOfThreads2() { return 256; }

  __host__ __device__ constexpr unsigned int GetNumberOfMAXBlocks() { return 256; }
} // End of ursus namespace
