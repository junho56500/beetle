#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <math.h>

using namespace std;

class Solution {
public:


    void reverseString(vector<char>& s) {
        int s_size = s.size();
        for (int i=0; i<int(s_size/2); i++)
        {
            char temp;
            temp = s[i];
            s[i] = s[s_size-1-i];
            s[s_size-1-i] = temp;
        }
    }

// Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.
// Assume the environment does not allow you to store 64-bit integers (signed or unsigned).

// Example 1:
// Input: x = 123
// Output: 321

// Example 2:
// Input: x = -123
// Output: -321

// Example 3:
// Input: x = 120
// Output: 21

    int reverse(int x) {
        bool minus = false;
        
        if(x<0)
        {
            minus = true;    
        }
        string s = to_string(x);
        
        int s_size = s.size();
        cout << s << endl;
        for (int i=0; i< int(s_size/2); i++)
        {
            char temp;
            temp = s[i];
            s[i] = s[s_size-i-1];
            s[s_size-i-1] = temp;
            cout << s << endl;
        }
        long long int lli = stoll(s);
        if((lli < -pow(2, 31)) || (lli > pow(2,31)-1))
        {
            return 0;
        }
        if(minus)
            return -lli;
        return lli;
    }

};


