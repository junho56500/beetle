#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

int bs(vector<int>& a, const int i)
{
    int ind_s = 0;
    int ind_e = a.size();
    sort(a.begin(), a.end());
    int ind_c = 0;

    while(ind_s < ind_e)
    {
        ind_c = int((ind_s + ind_e)/2);
        if (i == a[ind_c])
        {
            return ind_c;
        }
        else if (i < a[ind_c])
        {
            ind_e = ind_c;
        }
        else if(i > a[ind_c])
        {
            ind_s = ind_c;
        }
    }
    return ind_c;
}

int main()
{
    vector<int> a = {0,1,4,8,3,2,3,6,7};

    int i = bs(a, 4);

    cout << i;

    return 0;
}