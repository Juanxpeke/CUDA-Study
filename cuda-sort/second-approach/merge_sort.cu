#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

__global__ 
void partialSort(const int D, const int N, int* A, int* sortedA)
{
  int k;
	int j;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float tmp = 0.0f;

  if (i >= N) return;

	for (j = 0; j < N; j++)
  {
    tmp = 0.0f;
    for (k = 0; k < N; k++)
    {
      tmp += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = tmp;
  }
}

// Function to merge two subarrays arr[l..m] and arr[m+1..r] into arr
void merge(int* arr, int l, int m, int r) {
  int n1 = m - l + 1;
  int n2 = r - m;

  // Create temporary arrays
  int* L = new int[n1];
  int* R = new int[n2];

  // Copy data to temporary arrays L[] and R[]
  for (int i = 0; i < n1; i++)
    L[i] = arr[l + i];
  for (int j = 0; j < n2; j++)
    R[j] = arr[m + 1 + j];

  // Merge the temporary arrays back into arr[l..r]
  int i = 0; // Initial index of first subarray
  int j = 0; // Initial index of second subarray
  int k = l; // Initial index of merged subarray

  while (i < n1 && j < n2) {
    if (L[i] <= R[j]) {
      arr[k] = L[i];
      i++;
    } else {
      arr[k] = R[j];
      j++;
    }
    k++;
  }

  // Copy the remaining elements of L[], if any
  while (i < n1) {
    arr[k] = L[i];
    i++;
    k++;
  }

  // Copy the remaining elements of R[], if any
  while (j < n2) {
    arr[k] = R[j];
    j++;
    k++;
  }

  delete[] L;
  delete[] R;
}

// Main function to implement merge sort
void mergeSort(int* arr, int l, int r, int d) {
  if (l < r) {
    // Find the middle point
    int m = l + (r - l) / 2;

    // Sort first and second halves
    if (m - l > d) {
      mergeSort(arr, l, m, d);
      mergeSort(arr, m + 1, r, d);
    }

    // Merge the sorted halves
    merge(arr, l, m, r);
  }
}

// Encapsulated merge sort function
void sort(int* arr, int N, int D) {
  mergeSort(arr, 0, N - 1, D);
}

// Function to generate N random elements
void generateRandomArray(int N, int* A) {

  // Seed the random number generator
  srand(time(nullptr));

  // Generate N random elements
  for (int i = 0; i < N; ++i) {
    A[i] = (rand() % 100); // Generating random numbers between 0 and 99
  }
}

int main() {
  int N = 1 << 24; // Number of elements
  int D = 16;
  int numThreads = N / D;

  int* A = new int[N];
  int* sortedA = new int[N];

  generateRandomArray(N, A);

	// Allocate device memory for arrays A and sortedA
	int *dA, *dSortedA;
	cudaMalloc((void**) &dA, N * sizeof(int));
	cudaMalloc((void**) &dSortedA, N * sizeof(int));
  
	// Transfer array A from host to device
	cudaMemcpy(dA, A, N * sizeof(int), cudaMemcpyHostToDevice);

  // Blocks of size 16
  int blockSize = 16;

  // Round up in case numThreads is not a multiple of blockSize
  int numBlocks = (numThreads + blockSize - 1) / blockSize;

  // Run kernel on N elements on the GPU
	partialSort<<<numBlocks, blockSize>>>(D, N, dA, dSortedA);

	// Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

	// Transfer partially sorted array from device to host
	cudaMemcpy(sortedA, dSortedA, N * sizeof(int), cudaMemcpyDeviceToHost);

  sort(sortedA, N, blockSize);
  
  bool sorted = true;

  for (int i = 0; i < N - 1; i++) {
    if (sortedA[i] > sortedA[i + 1]) {
      sorted = false;
      break;
    }
  }
  
  if (sorted) {
    std::cout << "Success, array sorted" << std::endl;
  } else {
    std::cout << "Error, array is not sorted" << std::endl;
  }

  return 0;
}