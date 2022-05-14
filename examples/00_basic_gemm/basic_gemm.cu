/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.

  This example has delibrately been kept similar to the basic_gemm example from cutass-1.3 to 
  highlight the minimum amount of differences needed to transition to cutlass-2.0.

  Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/cus.h"

using namespace tensorflow;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
template<typename T>
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  T alpha,
  T const *A,
  int lda,
  T const *B,
  int ldb,
  T beta,
  T *C,
  int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<T,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  T,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  T,        // Data-type of C matrix
                                                  ColumnMajor>; // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  typename CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //
  
  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
template<typename T>
__global__ void InitializeMatrix_kernel(
  T *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = static_cast<T>(value);
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
template<typename T>
cudaError_t InitializeMatrix(T *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<T><<< grid, block >>>(matrix, rows, columns, seed);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
template<typename T>
cudaError_t AllocateMatrix(T **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(T) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix<T>(*matrix, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
template<typename T>
std::vector<T> TestCutlassGemm(int M, int N, int K, T alpha, T beta) {
  cudaError_t result;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Compute size in bytes of the C matrix.
  size_t sizeof_C = sizeof(T) * ldc * N;

  // Define pointers to matrices in GPU device memory.
  T *A;
  T *B;
  T *C;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix<T>(&A, M, K, 0);

  result = AllocateMatrix<T>(&B, K, N, 17);

  result = AllocateMatrix<T>(&C, M, N, 101);

  //
  // Launch CUTLASS GEMM.
  //

  result = CutlassSgemmNN<T>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  // Copy to host and verify equivalence.
  std::vector<T> host_cutlass(ldc * N);

  result = cudaMemcpy(host_cutlass.data(), C, sizeof_C, cudaMemcpyDeviceToHost);

  //
  // Free device memory allocations.
  //

  cudaFree(C);
  cudaFree(B);
  cudaFree(A);

  return host_cutlass;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[3] = { 512, 512, 512 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);
    ss >> scalars[i - 4];
  }

  //
  // Run the CUTLASS GEMM test.
  //

  auto host_cutlass = TestCutlassGemm<float>(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1]      // beta
  );

  auto host_reference = TestCutlassGemm<cus>(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    static_cast<cus>(scalars[0]),     // alpha
    static_cast<cus>(scalars[1])      // beta
  );

  bool flag = true;
  for (int i=0; i<host_cutlass.size(); i++) {
      flag &= (host_cutlass[i] == host_reference[i]);
  }

  if (!flag) {
    std::cerr << "CUTLASS results incorrect." << std::endl;

    return cudaErrorUnknown;
  }

  std::cout << "Passed." << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
