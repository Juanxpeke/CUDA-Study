#include <iostream>
#include <math.h>

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
        // C(i, j) = sum(over k) A(i, k) * B(k, j)
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
}

int main(void)
{
  // Matrices of 1K X 1K elements
  int N = 1 << 10;

  float* A = new float[N * N];
  float* B = new float[N * N];
  float* C = new float[N * N];

  // Initialize A and B matrices on the host
  for (int i = 0; i < N * N; i++)
  {
    A[i] = 1.0f;
    B[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  matMul(N, A, B, C);

  // Check for errors (all values should be 2048.0f)
  float maxError = 0.0f;

  for (int i = 0; i < N * N; i++)
  {
    maxError = fmax(maxError, fabs(C[i] - 2048.0f));
  }

  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete [] A;
  delete [] B;
  delete [] C;

  return 0;
}