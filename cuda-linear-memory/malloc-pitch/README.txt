Basic code using CUDA mallocPitch for 2D virtual matrices logic

For compiling

nvcc malloc_pitch.cu -o malloc_pitch

For running

.\malloc_pitch.exe

For profiling (doesn't work on VS terminal)

nvprof .\malloc_pitch.exe