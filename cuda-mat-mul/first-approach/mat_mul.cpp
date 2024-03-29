#include <iostream>
#include <algorithm>
#include <cmath>
#include "../mat_mul_defines.h"

void matMul(int N, float* A, float* B, float* C)
{
  int i, j, k;
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      C[i * N + j] = 0.0f;
      for (k = 0; k < N; k++)
      {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
}

int main(int argc, char* argv[])
{
  int N;
  if (argc == 2)
  {
    N = std::atoi(argv[1]);
  }
  else
  {
    N = 1024;
  }

  std::cout << "Running multiplication with N = " << N << std::endl;

  float* A = new float[N * N];
  float* B = new float[N * N];
  float* C = new float[N * N];

  // Initialize A and B matrices on the host (CPU)
  for (int i = 0; i < N * N; i++)
  {
    A[i] = A_VALUES;
    B[i] = B_VALUES;
  }

  // Run function on N * N elements on the CPU
  matMul(N, A, B, C);

  float maxError = 0.0f;

  for (int i = 0; i < N * N; i++)
  {
    maxError = std::max(maxError, std::fabs(C[i] - C_VALUES(N)));
  }

  if (maxError > EPSILON)
  {
    std::cout << "Error in multiplication, error value is " << maxError << std::endl;
  }
  else
  {
    std::cout << "Multiplication completed successfully" << std::endl;
  }

  // Free memory
  delete [] A;
  delete [] B;
  delete [] C;

  return 0;
}