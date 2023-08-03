Approach with the GPU, using 256 threads on N / 256 blocks

This approach use each thread to calculate a single row of the result C matrix, 
but does not use private memory yet, it keeps using global memory

For compiling

nvcc mat_mul.cu -o mat_mul_cuda

For running

.\mat_mul_cuda.exe

For profiling (doesn't work on VS terminal)

nvprof .\mat_mul_cuda.exe