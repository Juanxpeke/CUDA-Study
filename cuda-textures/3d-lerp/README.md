### 3D Lerp

cuda memory copy 3D function asks you a source and a destination

both can be either a cudaArray or a cudaPitchedPtr

cudaPitchedPtr is just a wrapper for a linear memory pointer which is known to be pitched (https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaPitchedPtr.html#structcudaPitchedPtr)