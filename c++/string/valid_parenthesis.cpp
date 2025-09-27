#include <iostream>
#include <stack>
#include <string>
#include <map>

class Solution {
public:
    bool isValid(std::string s) {
        std::stack<char> st;
        std::map<char, char> bracketMap = {
            {')', '('},
            {'}', '{'},
            {']', '['}
        };

        for (char c : s) {
            if (c == '(' || c == '{' || c == '[') {
                // If it's an opening bracket, push it onto the stack
                st.push(c);
            } else if (c == ')' || c == '}' || c == ']') {
                // If it's a closing bracket
                if (st.empty() || st.top() != bracketMap[c]) {
                    // Check if stack is empty or the top doesn't match
                    return false;
                }
                // If it matches, pop the opening bracket
                st.pop();
            }
        }
        // After checking all characters, the stack must be empty for a valid string
        return st.empty();
    }
};

int main() {
    Solution sol;
    
    std::string s1 = "()[]{}";
    std::string s2 = "(]";
    std::string s3 = "([)]";
    std::string s4 = "{[]}";
    
    std::cout << s1 << " is valid: " << (sol.isValid(s1) ? "true" : "false") << std::endl;
    std::cout << s2 << " is valid: " << (sol.isValid(s2) ? "true" : "false") << std::endl;
    std::cout << s3 << " is valid: " << (sol.isValid(s3) ? "true" : "false") << std::endl;
    std::cout << s4 << " is valid: " << (sol.isValid(s4) ? "true" : "false") << std::endl;

    return 0;
}