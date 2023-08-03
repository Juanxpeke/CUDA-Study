#include <iostream>
#include <math.h>

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


int main(void)
{
  // Matrices of 16K X 16K elements
  int N = 1 << 14;

  float* A = new float[N * N];
  float* B = new float[N * N];
  float* C = new float[N * N];

  // Initialize A and B matrices on the host
  for (int i = 0; i < N * N; i++)
  {
    A[i] = 1.0f;
    B[i] = 2.0f;
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

  // Run kernel on 1M elements on the GPU
	matMul<<<gridDim, blockDim>>>(N, dA, dB, dC);

	// Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

	// Transfer matrix C from device to host
	cudaMemcpy(C, dC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

  // Check for errors (all values should be 32768.0f)
  float maxError = 0.0f;

  for (int i = 0; i < N * N; i++)
  {
    maxError = fmax(maxError, fabs(C[i] - 32768.0f));
  }

  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete [] A;
  delete [] B;
  delete [] C;

  return 0;
}