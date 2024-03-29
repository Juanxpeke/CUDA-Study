# Matrix multiplication (GPU)

Approach with the GPU, using 16 x 16 threads on (N x N) / (16 x 16) blocks

This approach uses each thread to calculate a single value of the result matrix

## Compiling

nvcc mat_mul.cu -o mat_mul_cuda

## Running

You can run the program specifying the size N of the matrix (default value is 1024)

- Windows: .\mat_mul_cuda.exe N
- Linux: ./mat_mul_cuda N

## Profiling

- Windows: nvprof .\mat_mul_cuda.exe N
- Linux: nvprof ./mat_mul_cuda N

NOTE: This might not work on VS terminal