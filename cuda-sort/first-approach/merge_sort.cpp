#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

// Function to merge two subarrays arr[l..m] and arr[m+1..r] into arr
void merge(std::vector<int>& arr, int l, int m, int r) {
  int n1 = m - l + 1;
  int n2 = r - m;

  // Create temporary arrays
  std::vector<int> L(n1), R(n2);

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
}

// Main function to implement merge sort
void mergeSort(std::vector<int>& arr, int l, int r) {
  if (l < r) {
    // Find the middle point
    int m = l + (r - l) / 2;

    // Sort first and second halves
    mergeSort(arr, l, m);
    mergeSort(arr, m + 1, r);

    // Merge the sorted halves
    merge(arr, l, m, r);
  }
}

// Encapsulated merge sort function
void sort(std::vector<int>& arr) {
  int n = arr.size();
  mergeSort(arr, 0, n - 1);
}

// Function to generate a vector of N random elements
std::vector<int> generateRandomVector(int N) {
  std::vector<int> randomVector;

  // Seed the random number generator
  srand(time(nullptr));

  // Generate N random elements and push them into the vector
  for (int i = 0; i < N; ++i) {
    randomVector.push_back(rand() % 100); // Generating random numbers between 0 and 99
  }

  return randomVector;
}

int main() {
  int N = 1 << 24; // Number of elements
  std::vector<int> arr = generateRandomVector(N);

  sort(arr);
  
  bool sorted = true;

  for (int i = 0; i < arr.size() - 1; i++) {
    if (arr[i] > arr[i + 1]) {
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