// Given a string s, find the first non-repeating character in it and return its index. If it does not exist, return -1.

// Example 1:
// Input: s = "leetcode"
// Output: 0
// Explanation:
// The character 'l' at index 0 is the first character that does not occur at any other index.

// Example 2:
// Input: s = "loveleetcode"
// Output: 2

// Example 3:
// Input: s = "aabb"
// Output: -1

#include <vector>
#include <math.h>
#include <string>

using namespace std;

class Solution {
public:
    int firstUniqChar(string s) {
        std::vector<int> char_counts(pow(10,5), 0);

        for (char c : s) {
            char_counts[static_cast<unsigned char>(c)]++;
        }

        for (int i = 0; i < s.length(); ++i) {
            if (char_counts[static_cast<unsigned char>(s[i])] == 1) {
                return i;
            }
        }
        return -1;
    }
};