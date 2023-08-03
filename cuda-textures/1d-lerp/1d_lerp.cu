// ==================================
// CUDA textures lerp for 1D textures
// ==================================

#include <iostream>
#include <math.h>

// Simple texture fetching kernel with float coordinates
__global__ 
void betweenKernel(float* deviceResult, cudaTextureObject_t texObj, int size)
{
  // Calculate global coordinates
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate texture coordinates. When using CUDA arrays, a texel at index idx occupies an entire region 
  // defined by [idx, idx + 1) (see https://www.reedbeta.com/blog/texture-gathers-and-coordinate-precision/
  // and https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching)
  // In this case, as we are using lerp, the position in this region is important for the interpolation
  float textIdx = idx == size - 1 ? idx : idx + 1.0f;

  // Read from texture and write to global memory
  if (idx < size)
  {
    deviceResult[idx] = tex1D<float>(texObj, textIdx);
  }
}

// Host code
int main()
{
  // Maximum width for a 1D texture reference bound to a CUDA array is 131072
  const int N = 1 << 17;

  // Blocks of size 16
  int blockSize = 16; 

  // Round up in case N is not a multiple of blockSize
  int numBlocks = (N + blockSize - 1) / blockSize;

  // Initial value
  float initValue = 1.0f;

  // ===============================================
  // Allocate and set some host data, and CUDA array
  // ===============================================

  float* hostValues = new float[N];

  for (int i = 0; i < N; ++i)
  {
    hostValues[i] = i % 2 == 0 ? initValue : 3 * initValue;
  }

  // Channel descriptor for texture values, 32 bits = 4 bytes for each float, type is float
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  // Create 1D CUDA array of length N
  cudaArray_t cuArray;
  cudaMallocArray(&cuArray, &channelDesc, N);

  // Copy data located at address hostValues in host linear memory to device CUDA array memory
  // (For explanation see https://forums.developer.nvidia.com/t/cudamemcpytoarray-is-deprecated/71385/10)
  cudaMemcpy2DToArray(cuArray, 0, 0, hostValues,  N * sizeof(float), N * sizeof(float), 1, cudaMemcpyHostToDevice);

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
  resDesc.resType = cudaResourceTypeArray;

  // Assign resource memory
  resDesc.res.array.array = cuArray;

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
  texDesc.filterMode = cudaFilterModeLinear;

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
  cudaFreeArray(cuArray);
  cudaFree(deviceResult);

  // Free host memory
  free(hostValues);
  free(hostResult);

  return 0;
}