Basic code of how CUDA textures interpolation works.

This approach uses texture configuration with filter mode set as lerp and a kernel
for fetching the values in between. Values are a mix between I and 3I, and coordinates
are i + 1/2, so result should be an array filled of 2I.

For compiling

nvcc 1d_lerp.cu -o 1d_lerp

For running

.\1d_lerp.exe

For profiling (doesn't work on VS terminal)

nvprof .\1d_lerp.exe