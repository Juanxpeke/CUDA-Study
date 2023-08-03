// ==========================================================
// Malloc method for allocating 1D memory used as a 2D matrix
// ==========================================================

#include <iostream>
#include <math.h>

// Simple column sum kernel
__global__
void sumColumnsKernel(float* deviceMatrix, float* deviceSum, int width, int height)
{
  // Calculate coordinates
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  deviceSum[idx] = 0.0f;

  // Set values
  if (idx < width)
  {
    for (int i = 0; i < height; i++)
    {
      deviceSum[idx] += deviceMatrix[idx + i * width];
    }
  }
}

// Simple row sum kernel
__global__
void sumRowsKernel(float* deviceMatrix, float* deviceSum, int width, int height)
{
  // Calculate coordinates
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  deviceSum[idx] = 0.0f;

  // Set values
  if (idx < height)
  {
    for (int i = 0; i < width; i++)
    {
      deviceSum[idx] += deviceMatrix[idx * width + i];
    }
  }
}

// Matrix sum kernel, in this case, each thread traverses the entire matrix
__global__
void matrixSumKernel(float* deviceMatrix, double* deviceSum, int width, int height)
{
  // Calculate coordinates
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  deviceSum[idx] = 0.0f;

  for (int r = 0; r < height; ++r)
  {
    for (int c = 0; c < width; ++c)
    {
      deviceSum[idx] += deviceMatrix[r * width + c];
    }
  }
}

// Host code
int main()
{
  // Input matrix of 16387 x 16387 dimensions
  const int width = (1 << 14) + 3;
  const int height = (1 << 14) + 3;

  // Number of threads that will traverse entire matrix for special matrix sum kernel
  const int threadsMatrixSum = (1 << 6);

  // Blocks of size 16
  int blockSize = 16; 

  // Round up in case sizes are not multiple of blockSize
  int numBlocksColumns = (width + blockSize - 1) / blockSize;
  int numBlocksRows = (height + blockSize - 1) / blockSize;
  int numBlocksMatrix = (threadsMatrixSum + blockSize - 1) / blockSize;

  // Initialize data in host memory
  float* hostMatrix = new float[width * height];

  for (int i = 0; i < width * height; ++i)
  {
    hostMatrix[i] = 1.0f;
  }
  
  // Copy to device memory
  float* deviceMatrix;
  cudaMalloc(&deviceMatrix, width * height * sizeof(float));
  cudaMemcpy(deviceMatrix, hostMatrix, width * height * sizeof(float), cudaMemcpyHostToDevice);

  // Result device memory, for special sum matrix, you have to use double precision float
  float* deviceSumColumns;
  cudaMalloc(&deviceSumColumns, width * sizeof(float));
  float* deviceSumRows;
  cudaMalloc(&deviceSumRows, height * sizeof(float));
  double* deviceSumMatrix;
  cudaMalloc(&deviceSumMatrix, threadsMatrixSum * sizeof(double));

  // Call columns kernel
  sumColumnsKernel<<<numBlocksColumns, blockSize>>>(deviceMatrix, deviceSumColumns, width, height);
  cudaDeviceSynchronize();

  // Call rows kernel
  sumRowsKernel<<<numBlocksRows, blockSize>>>(deviceMatrix, deviceSumRows, width, height);
  cudaDeviceSynchronize();

  // Call matrix special kernel
  matrixSumKernel<<<numBlocksMatrix, blockSize>>>(deviceMatrix, deviceSumMatrix, width, height);
  cudaDeviceSynchronize();

  // Result host memory, for special sum matrix, you have to use double precision float
  float* hostSumColumns = new float[width];
  cudaMemcpy(hostSumColumns, deviceSumColumns, width * sizeof(float), cudaMemcpyDeviceToHost);
  float* hostSumRows = new float[height];
  cudaMemcpy(hostSumRows, deviceSumRows, height * sizeof(float), cudaMemcpyDeviceToHost);
  double* hostSumMatrix = new double[threadsMatrixSum];
  cudaMemcpy(hostSumMatrix, deviceSumMatrix, threadsMatrixSum * sizeof(double), cudaMemcpyDeviceToHost);

  // Check for errors (all values should be height, width, or width * height, respectively)
  float maxErrorColumns = 0.0f;
  float maxErrorRows = 0.0f;
  float maxErrorMatrix = 0.0f;

  for (int i = 0; i < width; i++)
  {
    maxErrorColumns = fmax(maxErrorColumns, fabs(hostSumColumns[i] - height));
  }

  for (int i = 0; i < height; i++)
  {
    maxErrorRows = fmax(maxErrorRows, fabs(hostSumRows[i] - width));
  }

  for (int i = 0; i < threadsMatrixSum; i++)
  {
    maxErrorMatrix = fmax(maxErrorMatrix, fabs(hostSumMatrix[i] - width * height));
  }

  std::cout << "Max error for columns: " << maxErrorColumns << std::endl;
  std::cout << "Max error for rows: " << maxErrorRows << std::endl;
  std::cout << "Max error for special matrix sum: " << maxErrorMatrix << std::endl;

  // Free device memory
  cudaFree(deviceMatrix);
  cudaFree(deviceSumColumns);
  cudaFree(deviceSumRows);
  cudaFree(deviceSumMatrix);

  // Free host memory
  free(hostMatrix);
  free(hostSumColumns);
  free(hostSumRows);
  free(hostSumMatrix);

  return 0;
}