#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Function to partition the array and return the pivot index
// Here, we choose the first element as the pivot
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[low]; // Choose the first element as the pivot
    int i = low + 1;      // Start from the element after the pivot

    for (int j = low + 1; j <= high; j++) {
        // If the current element is smaller than the pivot
        if (arr[j] < pivot) {
            swap(arr[i], arr[j]);
            i++;
        }
    }
    // Place the pivot in its correct sorted position
    swap(arr[low], arr[i - 1]);
    return (i - 1);
}

// The main quicksort function
void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        // pi is the partitioning index
        int pi = partition(arr, low, high);

        // Separately sort elements before and after the partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

void quick(vector<int>& arr, int low, int high)
{
    if (low < high)
    {    
        int piv = arr[low];
        int i = low + 1;

        for (int j = low + 1; j <= high; j++)
        {
            if (arr[j] < piv)
            {
                swap(arr[i], arr[j]);
                i++;
            }
        }
        int pi = i - 1;
        swap(arr[pi], arr[low]);

        quick(arr, low, pi - 1);
        quick(arr, pi + 1, high);
    }
}

int main() {
    std::vector<int> data = {10, 7, 8, 9, 1, 5};
    int n = data.size();

    std::cout << "Original array: ";
    for (int x : data) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // quickSort(data, 0, n - 1);
    quick(data, 0, n - 1);

    std::cout << "Sorted array: ";
    for (int x : data) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    return 0;
}
