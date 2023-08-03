Basic code of how to use linear memory with CUDA textures.

This approach uses default texture configuration and a kernel for duplicating the array values by fetching
from the texture.

For compiling

nvcc linear_memory_texture.cu -o linear_memory_texture

For running

.\linear_memory_texture.exe

For profiling (doesn't work on VS terminal)

nvprof .\linear_memory_texture.exe