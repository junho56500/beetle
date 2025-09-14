#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

void selection_sort(vector<int>& arr)
{
    int n = arr.size();

    for (int i=0; i<n; i++)
    {
        int idx_min = i;
        for (int j=i+1; j<n; j++)
        {
            if (arr[j] < arr[idx_min])
            {
                idx_min = j;
            }
        }
        swap(arr[i], arr[idx_min]);
    }
}

int main()
{
    vector<int> arr = {4,2,6,8,1,9,0,10,19,25,5};
    selection_sort(arr);

    for (int i =0 ; i<arr.size(); i++)
    {
        cout << arr[i];
    }
    return 0;
}


void selection(vector<int>& arr)
{
    for(int i=0; i<sizeof(arr); i++)
    {
        int min_idx = i;
        for(int j=i+1; j<sizeof(arr); j++)
        {
            if(arr[min_idx]<arr[j])
            {
                min_idx = j;
            }
        }
        swap(arr[min_idx], arr[i]);
    }
}