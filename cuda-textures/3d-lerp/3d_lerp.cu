// ==================================
// CUDA textures lerp for 1D textures
// ==================================

#include <iostream>
#include <math.h>

// Simple texture fetching kernel with float coordinates
__global__ 
void betweenKernel(float* output, cudaTextureObject_t texObj, int size)
{
  // Calculate global coordinates
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate texture coordinates, that is, idx + 1/2 only if idx is not last index
  float textIdx = idx == size - 1 ? idx - 0.5f : idx + 0.5f;
  
  // Weird CUDA offset (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching)
  textIdx += 0.5f;

  // Read from texture and write to global memory
  if (idx < size)
  {
    output[idx] = tex1D<float>(texObj, textIdx);
  }
}

// Host code
int main()
{
  // Maximum dimension length for a 3D texture reference bound to a CUDA array is 16384, but we'll use 4096
  const int N = 1 << 12;

  // Blocks of size 4 x 4 x 4
  int blockSize = 4; 

  // Round up in case N is not a multiple of blockSize
  int numBlocks = (N + blockSize - 1) / blockSize;

	dim3 blockDim(blockSize, blockSize, blockSize);
	dim3 gridDim(numBlocks, numBlocks, numBlocks);

  // Initial value
  float initValue = 1.0f;

  // ===============================================
  // Allocate and set some host data, and CUDA array
  // ===============================================

  float* hostValues = new float[N];

  for (int i = 0; i < N * N * N; ++i)
  {
    hostValues[i] = i % 2 == 0 ? initValue : 3 * initValue;
  }

  // Channel descriptor for texture values, 32 bits = 4 bytes for each float, type is float
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  // CUDA array dimensions
  cudaExtent cuArrayExtent = { N, N, N };

  // Create 3D CUDA array of N x N x N
  cudaArray_t cuArray;
  cudaMalloc3DArray(&cuArray, &channelDesc, cuArrayExtent);

  // CUDA memory copy parameters
  cudaMemcpy3DParms cuMemcpyParms = {0};

  cuMemcpyParms.dstArray = cuArray;

  // Copy data located in host memory to device CUDA array memory
  cudaMemcpy3D(&cuMemcpyParms);

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
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.addressMode[2] = cudaAddressModeClamp;

  // The filtering mode to be used when fetching from the texture
  // IGNORED if cudaResourceDesc::resType is cudaResourceTypeLinear
  texDesc.filterMode = cudaFilterModeLinear;

  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  // Allocate result of duplication in device memory
  float* output;
  cudaMalloc(&output, N * sizeof(float));

  // Allocate result of duplication in host memory
  float* result = new float[N];

  // Invoke kernel
  betweenKernel<<<numBlocks, blockSize>>>(output, texObj, N);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Copy data from device back to host
  cudaMemcpy(result, output, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Check for errors (all values should be 2 * initValue)
  float maxError = 0.0f;

  for (int i = 0; i < N; i++)
  {
    maxError = fmax(maxError, fabs(result[i] - 2 * initValue));
  }

  std::cout << "Max error: " << maxError << std::endl;

  // Destroy texture object
  cudaDestroyTextureObject(texObj);

  // Free device memory
  cudaFreeArray(cuArray);
  cudaFree(output);

  // Free host memory
  free(hostValues);
  free(result);

  return 0;
}