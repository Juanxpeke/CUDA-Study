Basic code using CUDA malloc for 2D virtual matrices logic

For compiling

nvcc malloc_for_2d.cu -o malloc_for_2d

For running

.\malloc_for_2d.exe

For profiling (doesn't work on VS terminal)

nvprof .\malloc_for_2d.exe