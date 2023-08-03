Approach with the GPU, using 256 threads on N / 256 blocks

For compiling

nvcc add.cu -o add_cuda

For running

.\add_cuda.exe

For profiling (doesn't work on VS terminal)

nvprof .\add_cuda.exe