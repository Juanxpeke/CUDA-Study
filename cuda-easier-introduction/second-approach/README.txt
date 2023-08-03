Approach with the GPU, but using only 1 thread

For compiling

nvcc add.cu -o add_cuda

For running

.\add_cuda.exe

For profiling (doesn't work on VS terminal)

nvprof .\add_cuda.exe