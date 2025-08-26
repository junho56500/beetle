#include <iostream>
#include <vector>
#include <algorithm> // Required for std::swap

// A helper function to print the contents of a vector.
void printVector(const std::vector<int>& arr) {
    for (int element : arr) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

// 1. Bubble Sort
// Time Complexity: O(n^2)
// Compares adjacent elements and swaps them if they are in the wrong order.
// The largest element "bubbles" to the end of the array with each pass.
void bubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        // Last i elements are already in place
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                // Swap if the element found is greater than the next element
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// 2. Selection Sort
// Time Complexity: O(n^2)
// Finds the minimum element in the unsorted part of the array
// and places it at the beginning of the unsorted part.
void selectionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        // Find the index of the minimum element in the remaining unsorted array
        int min_idx = i;
        for (int j = i + 1; j < n; ++j) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        // Swap the found minimum element with the first element of the unsorted part
        std::swap(arr[min_idx], arr[i]);
    }
}

// 3. Insertion Sort
// Time Complexity: O(n^2)
// Builds the final sorted array one item at a time by repeatedly taking
// the next element from the unsorted part and inserting it into the
// correct position in the sorted part.
void insertionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 1; i < n; ++i) {
        int key = arr[i];
        int j = i - 1;

        // Move elements of arr[0..i-1] that are greater than key,
        // to one position ahead of their current position.
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

// 4. Quick Sort
// Time Complexity: Average O(n log n), Worst-case O(n^2)
// A divide and conquer algorithm. It picks an element as a pivot and partitions
// the given array around the picked pivot.
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high]; // Choosing the last element as the pivot
    int i = (low - 1);     // Index of smaller element

    for (int j = low; j < high; ++j) {
        if (arr[j] <= pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return (i + 1);
}

void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    // Test Bubble Sort
    std::vector<int> bubble_arr = {64, 34, 25, 12, 22, 11, 90};
    std::cout << "Original array for Bubble Sort: ";
    printVector(bubble_arr);
    bubbleSort(bubble_arr);
    std::cout << "Sorted array (Bubble Sort): ";
    printVector(bubble_arr);
    std::cout << std::endl;

    // Test Selection Sort
    std::vector<int> selection_arr = {64, 25, 12, 22, 11, 90, 34};
    std::cout << "Original array for Selection Sort: ";
    printVector(selection_arr);
    selectionSort(selection_arr);
    std::cout << "Sorted array (Selection Sort): ";
    printVector(selection_arr);
    std::cout << std::endl;

    // Test Insertion Sort
    std::vector<int> insertion_arr = {12, 11, 13, 5, 6};
    std::cout << "Original array for Insertion Sort: ";
    printVector(insertion_arr);
    insertionSort(insertion_arr);
    std::cout << "Sorted array (Insertion Sort): ";
    printVector(insertion_arr);
    std::cout << std::endl;
    
    // Test Quick Sort
    std::vector<int> quick_arr = {10, 7, 8, 9, 1, 5};
    std::cout << "Original array for Quick Sort: ";
    printVector(quick_arr);
    quickSort(quick_arr, 0, quick_arr.size() - 1);
    std::cout << "Sorted array (Quick Sort): ";
    printVector(quick_arr);
    std::cout << std::endl;

    return 0;
}