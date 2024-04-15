# Second Approach

Approach with the GPU, but using only 1 thread

## Compiling

nvcc add.cu -o add_cuda

## Running

- Linux: ./add_cuda
- Windows:  .\add_cuda.exe

## Profiling (doesn't work on Visual Studio Code terminal)

- Linux: nvprof ./add_cuda
- Windows: nvprof .\add_cuda.exe