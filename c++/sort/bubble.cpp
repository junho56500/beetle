#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

void bubbleSort(vector<int>& arr)
{
    for (int i=0; i<arr.size()-1; i++)
    {
        for (int j =0; j<arr.size()-i-1; j++)
        {
            if (arr[i] > arr[j])
            {
                swap(arr[j], arr[i]);
            }
        }
    }   
}

int main()
{
    vector<int> arr = {4,2,6,8,1,9,4,6,0};
    bubbleSort(arr);

    for (int i =0 ; i<arr.size(); i++)
    {
        cout << i;
    }
    return 0;
}