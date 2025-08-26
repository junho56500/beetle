#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

void insertionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 1; i < n; ++i) {
        int key = arr[i];
        int j = i - 1;

        // The while loop correctly shifts elements to the right to make space for the key.
        // The key is then inserted once, after the shifting is complete.
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

int main()
{
    vector<int> arr = {4,2,6,8,1,9,0,10,19,25,5};
    insertionSort(arr);

    for (int i =0 ; i<arr.size(); i++)
    {
        cout << arr[i];
    }
    return 0;
}