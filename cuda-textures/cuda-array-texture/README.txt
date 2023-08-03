Basic code of how to use CUDA arrays with CUDA textures.

This approach uses default texture configuration and a kernel for duplicating the array values by fetching
from the texture.

For compiling

nvcc cuda_array_texture.cu -o cuda_array_texture

For running

.\cuda_array_texture.exe

For profiling (doesn't work on VS terminal)

nvprof .\cuda_array_texture.exe