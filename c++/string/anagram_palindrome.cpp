#include <iostream>
#include <string>
#include <vector>
#include <math.h>

using namespace std;

// Given two strings s and t, return true if t is an anagram of s, and false otherwise.

// Example 1:
// Input: s = "anagram", t = "nagaram"
// Output: true

// Example 2:
// Input: s = "rat", t = "car"
// Output: false
 
class Solution {
public:
    bool isAnagram(string s, string t) {
        vector<int> countS(5*pow(10,4), 0);
        vector<int> countT(5*pow(10,4), 0);
        for (const auto& c : s)
        {
            countS[static_cast<unsigned int>(c)]++;
        }
        for (const auto& c : t)
        {
            countT[static_cast<unsigned int>(c)]++;
        }
        for(int i = 0; i<countS.size(); i++ )
        {
            if(countS[i] != countT[i])
            {
                return false;
            }
        }
        return true;
    }
};

