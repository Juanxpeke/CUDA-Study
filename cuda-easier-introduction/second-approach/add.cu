#include <iostream>
#include <math.h>

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
  {
    y[i] = x[i] + y[i];
  }
}

int main(void)
{
  // Arrays of 16M elements
  int N = 1 << 24;

  // Allocate Unified Memory -- accessible from CPU or GPU (https://developer.nvidia.com/blog/unified-memory-in-cuda-6/)
  float *x, *y;
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // Initialize x and y arrays on the host
  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 16M elements on the GPU, this function launches one GPU thread to run add
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;

  for (int i = 0; i < N; i++)
  {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  }

  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}