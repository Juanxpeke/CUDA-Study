#include <iostream>
#include <algorithm>
#include <cmath>
#include "../mat_mul_defines.h"

__global__ 
void matMul(const int N, float* A, float* B, float* C)
{
  int k;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  float tmp = 0.0f;

	if (i >= N || j >= N) return;
  
  for (k = 0; k < N; k++)
  {
    // Too many accesses to global memory
    tmp += A[i * N + k] * B[k * N + j];
  }

  C[i * N + j] = tmp;
}


int main(int argc, char* argv[])
{
  int N;
  if (argc == 2)
  {
    N = std::atoi(argv[1]);
  }
  else
  {
    N = 1024;
  }

  std::cout << "Running multiplication with N = " << N << std::endl;

  float* A = new float[N * N];
  float* B = new float[N * N];
  float* C = new float[N * N];

  // Initialize A and B matrices on the host
  for (int i = 0; i < N * N; i++)
  {
    A[i] = A_VALUES;
    B[i] = B_VALUES;
  }

	// Allocate device memory for matrices A, B, and C
	float *dA, *dB, *dC;
	cudaMalloc((void**) &dA, N * N * sizeof(float));
	cudaMalloc((void**) &dB, N * N * sizeof(float));
	cudaMalloc((void**) &dC, N * N * sizeof(float));

	// Transfer matrices A and B from host to device
	cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

  // Blocks of size 16 x 16
  int blockSize = 16;

  // Round up in case N is not a multiple of blockSize
  int numBlocks = (N + blockSize - 1) / blockSize;

	// Define block and grid dimensions
	dim3 blockDim(blockSize, blockSize);
	dim3 gridDim(numBlocks, numBlocks);

  // Run kernel on N * N elements on the GPU
	matMul<<<gridDim, blockDim>>>(N, dA, dB, dC);

	// Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

	// Transfer matrix C from device to host
	cudaMemcpy(C, dC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

  // Check for errors
  float maxError = 0.0f;

  for (int i = 0; i < N * N; i++)
  {
    maxError = std::max(maxError, std::fabs(C[i] - C_VALUES(N)));
  }

  if (maxError > EPSILON)
  {
    std::cout << "Error in multiplication, error value is " << maxError << std::endl;
  }
  else
  {
    std::cout << "Multiplication completed successfully" << std::endl;
  }

  // Free host memory
  delete [] A;
  delete [] B;
  delete [] C;

  return 0;
}