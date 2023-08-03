// =====================================================================================================
// Linear memory does not support normalized coordinates, so changing address and filter is not possible
// =====================================================================================================

#include <iostream>
#include <math.h>

// Simple manual parallel setter
__global__
void setKernel(float* deviceValues, float value, int size)
{
  // Calculate coordinates
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Set values
  if (idx < size)
  {
    deviceValues[idx] = value;
  }
}

// Simple duplication kernel that uses texture fetching
__global__ 
void betweenKernel(float* deviceResult, cudaTextureObject_t texObj, int size)
{
  // Calculate coordinates
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Read from texture and write to global memory
  if (idx < size)
  {
    deviceResult[idx] = 2 * tex1Dfetch<float>(texObj, idx);
  }
}

// Host code
int main()
{
  // Maximum width for a 1D texture reference bound to linear memory is 2 ^ 28 
  const int N = 1 << 28;

  // Blocks of size 16
  int blockSize = 16; 

  // Round up in case N is not a multiple of blockSize
  int numBlocks = (N + blockSize - 1) / blockSize;

  // Initial value
  float initValue = 1.0f;

  // ==============================
  // Allocate and set device memory
  // ==============================
  float* deviceValues;
  cudaMalloc(&deviceValues, N * sizeof(float));
  setKernel<<<numBlocks, blockSize>>>(deviceValues, initValue, N);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  // Channel descriptor for texture values, 32 bits = 4 bytes for each float, type is float
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  // ========================
  // Specify texture resource
  // ========================

  // Describes the data to texture from
  struct cudaResourceDesc resDesc;

  // Used to initialize all elements of the structures to zero. If you don't do this, elements of
  // the structure may have random values
  memset(&resDesc, 0, sizeof(resDesc));

  // There are two types of device memory. Linear memory and CUDA arrays CUDA arrays are optimized
  // memory for texture fetching
  resDesc.resType = cudaResourceTypeLinear;

  // Assign resource memory
  resDesc.res.linear.devPtr = deviceValues;
  resDesc.res.linear.sizeInBytes = N * sizeof(float);
  resDesc.res.linear.desc = channelDesc;

  // =================================
  // Specify texture object parameters
  // =================================

  // Describes how the data should be sampled
  struct cudaTextureDesc texDesc;
  
  // Used to initialize all elements of the structures to zero. If you don't do this, elements of 
  // the structure may have random values
  memset(&texDesc, 0, sizeof(texDesc));
  
  // The read mode, which is equal to cudaReadModeNormalizedFloat or cudaReadModeElementType. It is 
  // used to determine if the returned value is normalized or not
  texDesc.readMode = cudaReadModeElementType;

  // Whether texture coordinates (access input) are normalized or not
  // IGNORED if cudaResourceDesc::resType is cudaResourceTypeLinear
  texDesc.normalizedCoords = false;

  // The addressing mode is specified as an array of size three whose first, second, and third elements 
  // specify the addressing mode for the first, second, and third texture coordinates, respectively;
  // cudaAddressModeWrap and cudaAddressModeMirror are only supported for normalized texture coordinates
  // IGNORED if cudaResourceDesc::resType is cudaResourceTypeLinear
  texDesc.addressMode[0] = cudaAddressModeClamp;

  // The filtering mode to be used when fetching from the texture
  // IGNORED if cudaResourceDesc::resType is cudaResourceTypeLinear
  texDesc.filterMode = cudaFilterModePoint;

  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  // Allocate result of duplication in device memory
  float* deviceResult;
  cudaMalloc(&deviceResult, N * sizeof(float));

  // Invoke kernel
  betweenKernel<<<numBlocks, blockSize>>>(deviceResult, texObj, N);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Allocate result of duplication in host memory
  float* hostResult = new float[N];
  cudaMemcpy(hostResult, deviceResult, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Check for errors (all values should be 2 * initValue)
  float maxError = 0.0f;

  for (int i = 0; i < N; i++)
  {
    maxError = fmax(maxError, fabs(hostResult[i] - 2 * initValue));
  }

  std::cout << "Max error: " << maxError << std::endl;

  // Destroy texture object
  cudaDestroyTextureObject(texObj);

  // Free device memory
  cudaFree(deviceValues);
  cudaFree(deviceResult);

  // Free host memory
  free(hostResult);

  return 0;
}