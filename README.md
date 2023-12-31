# CUDA Study

Simple and personal repository for documenting CUDA features and GPU parallel algorithms with code samples. The code provided here is using CUDA Toolkit v12.2 (2023), and should work on any OS. However, the build instructions and executable files are intended for a Windows environment.

## How this repository is structured

Each folder within this repository has code about an specific topic (Introduction, Linear Memory, Textures, etc.) and a README with basic explanation about the code logic.

The code for each topic is inside subfolders containing different features or approaches related to the same topic. In addition to the code, there is also an executable ```executable.exe``` for running the program and a README with instructions.

For subfolders that represent different approaches to the same problem, there are files ```profile.txt``` that show its performance, calculated using Nvprof (NVIDIA's profiler), on a computer with a NVIDIA GTX 1660 GPU, Intel Core i5-8600K CPU and 16GB of RAM.  

## Setup

As a general requirement for all the topics, you need to install the NVIDIA CUDA Toolkit (https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

## Basics

CUDA is a **parallel programming model** for general purpose computing on the NVIDIA GPUs. It allows you to run **data parallel** logic in your GPU threads for **HPC**.

For an introduction to the CUDA logic, see 
 
- An Even Easier Introduction to CUDA (https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

For simple information of almost any CUDA specific topic, see

- CUDA C Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- CUDA Runtime API (https://docs.nvidia.com/cuda/cuda-runtime-api/)

## Linear memory

Linear memory is the most basic memory layout used by CUDA, it is represented as a contiguous memory chunk.

Pitched memory is padded linear memory, that can be more efficient due to word alignment and other stuff I don't understand at the moment.

### Sources

- CUDA C Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory)
- CUDA Runtime API (https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY)
- How and when should I use pitched memory (https://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api)
- cudaMallocPitch and cudaMemcpy2D (https://stackoverflow.com/questions/35771430/cudamallocpitch-and-cudamemcpy2d)

## Shared Memory

Shared memory is an space of memory shared by a group of threads belonging to the same CUDA block. It can be used to

- Apply shared logic between threads
- Avoid reading from global memory
- Avoid using local memory when thread data is too large

### Sources

- Using shared memory in CUDA (https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- CUDA C Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- CUDA shared memory webinar (https://developer.download.nvidia.com/CUDA/training/sharedmemoryusage_july2011.mp4)
- CUDA GPU memory (https://www.ce.jhu.edu/dalrymple/classes/602/Class13.pdf)
- Local Memory and Register Spilling (https://developer.download.nvidia.com/CUDA/training/register_spilling.pdf)
- Why is padding for shared memory required? (https://stackoverflow.com/questions/15056842/when-is-padding-for-shared-memory-really-required)
- Advanced CUDA webinar (https://developer.download.nvidia.com/CUDA/training/NVIDIA_GPU_Computing_Webinars_CUDA_Memory_Optimization.pdf)

## Texture Memory

Special type of memory that can be more efficient than global memory under certain situations. It also has unique features like filtering (hardware-implemented linear, bilinear, and trilinear interpolation), out-of-range fetching, or normalized coordinates.

In order to use texture memory, CUDA provides a struct called texture object, which is then passed to kernels as an argument. There are two ways for creating texture objects, with linear memory, or using CUDA arrays.

### Cuda Arrays

CUDA arrays are opaque memory layouts optimized for texture fetching.

### Sources

- CUDA C Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)
- CUDA Runtime API (https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT)
- CUDA Channel Format Descriptor (http://cuda-programming.blogspot.com/2013/02/cudachannelformatdesc-in-cuda-how-to.html)
- CUDA Arrays, old API (http://cuda-programming.blogspot.com/2013/02/cuda-array-in-cuda-how-to-use-cuda.html)
- Advanced CUDA webinar (https://developer.download.nvidia.com/CUDA/training/NVIDIA_GPU_Computing_Webinars_CUDA_Memory_Optimization.pdf)
- Textures float coordinates logic (https://www.reedbeta.com/blog/texture-gathers-and-coordinate-precision/)

## OpenGL Interoperability

With CUDA, you can modify OpenGL vertex buffers data using parallel computing.

### Sources

- CUDA C Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#opengl-interoperability)
- CUDA Runtime API (https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html)

