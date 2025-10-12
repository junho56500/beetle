#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <transform>

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

// A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.
// Given a string s, return true if it is a palindrome, or false otherwise.

// Example 1:
// Input: s = "A man, a plan, a canal: Panama"
// Output: true
// Explanation: "amanaplanacanalpanama" is a palindrome.

// Example 2:
// Input: s = "race a car"
// Output: false
// Explanation: "raceacar" is not a palindrome.

// Example 3:
// Input: s = " "
// Output: true
// Explanation: s is an empty string "" after removing non-alphanumeric characters.
// Since an empty string reads the same forward and backward, it is a palindrome.

class Solution {
public:
    bool isPalindrome(string s) {
        string a;
        for(const auto& c  :s)
        {
            if(isalpha(c) || isdigit(c))
            {
                a.push_back(c);
            }
        }
        transform(a.begin(), a.end(), a.begin(), ::tolower);
        
        cout << a;
        for(int i = 0; i<int(a.size()/2); i++)
        {
            if(a[i] != a[a.size()-1-i])
            {
                return false;
            }
        }
        return true;
    }
};