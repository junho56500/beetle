#include "stdio.h"
#include <vector>
#include <string>
#include <iostream> // For output
#include <algorithm> // For std::sort, std::find, etc.

using namespace std;

int main() {
    std::string s1;                     // Empty string
    std::string s2 = "Hello";           // From a C-style string literal
    std::string s3("World");            // From a C-style string literal (constructor)
    std::string s4(s2);                 // Copy constructor
    std::string s5(s2, 1, 3);           // Substring of s2: "ell" (pos 1, length 3)
    std::string s6(5, 'A');             // Five 'A' characters: "AAAAA"
    std::string s7 = s2 + " " + s3;     // Concatenation
    std::string s8 {'C', '+', '+'};     // Initializer list (C++11+)

    char c = s2[0]; // 'H'
    s2[0] = 'h';    // s2 becomes "hello"

    try {
        char c_safe = s2.at(0);
        char c_error = s2.at(100); // Throws exception if s2.length() < 101
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    if (!s2.empty()) {
        char& first = s2.front(); // 'h'
    }

    if (!s2.empty()) {
        char& last = s2.back(); // 'o'
    }

    std::string str = "example";
    char firstChar = str[0]; // 'e'
    str[0] = 'E';            // str becomes "Example"

    try {
        std::string str = "test";
        char c = str.at(1); // 'e'
        char invalid = str.at(10); // Throws std::out_of_range
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    if (!str.empty()) {
        char& f = str.front();
    }

    if (!str.empty()) {
        char& b = str.back();
    }

    const char* c_str_ptr = str.c_str();
    printf("C-style string: %s\n", c_str_ptr);

    std::cout << "Length: " << str.length() << std::endl;

    std::string message = "Hello";
    message += " World"; // message is "Hello World"
    message += '!';      // message is "Hello World!"

    std::string s = "ABC";
    s.append("DEF");           // s is "ABCDEF"
    s.append(3, 'X');          // s is "ABCDEFXXX"
    s.append("12345", 2);      // s is "ABCDEFXXX12" (appends "12")

    std::string s = "old";
    s.assign("new content"); // s is "new content"
    s.assign(5, 'Z');        // s is "ZZZZZ"

    std::string text = "World!";
    text.insert(0, "Hello "); // text is "Hello World!"

    std::string text = "Heloo World";
    text.erase(2, 1); // text is "Hello World" (removes 'o' at index 2)

    return 0;
}