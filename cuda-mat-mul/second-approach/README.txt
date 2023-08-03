Approach with the GPU, using 16 x 16 threads on (N x N) / (16 x 16) blocks

This approach use each thread to calculate a single value of the result C matrix

For compiling

nvcc mat_mul.cu -o mat_mul_cuda

For running

.\mat_mul_cuda.exe

For profiling (doesn't work on VS terminal)

nvprof .\mat_mul_cuda.exe