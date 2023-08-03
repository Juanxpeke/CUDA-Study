#include <iostream>
#include <math.h>

__global__ 
void matMul(const int N, float* A, float* B, float* C)
{
  __shared__ float Bwrk[1 << 14];

  int k;
	int j;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int iloc = threadIdx.x;
  int nloc = blockDim.x;
  float tmp = 0.0f;
  float Awrk[1 << 14];

  if (i >= N) return;
  
  for (k = 0; k < N; k++)
  {
    Awrk[k] = A[i * N + k];
  }

	for (j = 0; j < N; j++)
  {
    // Sincronizacion para almacenar la columna. Si son t threads por grupo que calcularian t filas, cada thread
    // se encarga de almacenar N / t valores de la columna actual en el arreglo local. Ej: Si son 3 threads por
    // grupo, el thread 0 de cada grupo se encarga de almacenar Bwrk[0], Bwrk[3], Bwrk[6], ...  
    for (k = iloc; k < N; k += nloc)
    {
      Bwrk[k] = B[k * N + j];
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    tmp = 0.0f;
    for (k = 0; k < N; k++)
    {
      tmp += Awrk[k] * Bwrk[k];
    }
    C[i * N + j] = tmp;

    // Synchronize (ensure the j iteration was done by all threads before passing to next interation)
    __syncthreads();
  }
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

  // Blocks of size 256
  int blockSize = 256;

  // Round up in case N is not a multiple of blockSize
  int numBlocks = (N + blockSize - 1) / blockSize;

  // Run kernel on 1M elements on the GPU
	matMul<<<numBlocks, blockSize>>>(N, dA, dB, dC);

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