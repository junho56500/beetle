#include "stdio.h"
#include <vector>
#include <string>
#include <iostream> // For output
#include <algorithm> // For sort, find, etc.
#include <sstream>

using namespace std;

int main() {
    string s1;                     // Empty string
    string s2 = "Hello";           // From a C-style string literal
    string s3("World");            // From a C-style string literal (constructor)
    string s4(s2);                 // Copy constructor
    string s5(s2, 1, 3);           // Substring of s2: "ell" (pos 1, length 3)
    string s6(5, 'A');             // Five 'A' characters: "AAAAA"
    string s7 = s2 + " " + s3;     // Concatenation
    string s8 {'C', '+', '+'};     // Initializer list (C++11+)

    char c = s2[0]; // 'H'
    s2[0] = 'h';    // s2 becomes "hello"

    try {
        char c_safe = s2.at(0);
        char c_error = s2.at(100); // Throws exception if s2.length() < 101
    } catch (const out_of_range& e) {
        cerr << "Error: " << e.what() << endl;
    }

    if (!s2.empty()) {
        char& first = s2.front(); // 'h'
    }

    if (!s2.empty()) {
        char& last = s2.back(); // 'o'
    }

    string str = "example";
    char firstChar = str[0]; // 'e'
    str[0] = 'E';            // str becomes "Example"

    try {
        string str = "test";
        char c = str.at(1); // 'e'
        char invalid = str.at(10); // Throws out_of_range
    } catch (const out_of_range& e) {
        cerr << "Error: " << e.what() << endl;
    }

    if (!str.empty()) {
        char& f = str.front();
    }

    if (!str.empty()) {
        char& b = str.back();
    }

    const char* c_str_ptr = str.c_str();
    printf("C-style string: %s\n", c_str_ptr);

    cout << "Length: " << str.length() << endl;

    string message = "Hello";
    message += " World"; // message is "Hello World"
    message += '!';      // message is "Hello World!"

    string s = "ABC";
    s.append("DEF");           // s is "ABCDEF"
    s.append(3, 'X');          // s is "ABCDEFXXX"
    s.append("12345", 2);      // s is "ABCDEFXXX12" (appends "12")

    string s = "old";
    s.assign("new content"); // s is "new content"
    s.assign(5, 'Z');        // s is "ZZZZZ"

    string text = "World!";
    text.insert(0, "Hello "); // text is "Hello World!"

    string text = "Heloo World";
    text.erase(2, 1); // text is "Hello World" (removes 'o' at index 2)     //strange....

    if (s1.compare(s2) == 0) { // Check if s1 and s2 are equal
        // ...
    }

    string sentence = "The quick brown fox";
    size_t pos = sentence.find("quick"); // pos will be 4
    if (pos != string::npos) {
        cout << "Found 'quick' at: " << pos << endl;
    }

    string full_name = "John Doe";
    string first_name = full_name.substr(0, 4); // "John" substr(position, count)
    string last_name = full_name.substr(5);    // "Doe" (from index 5 to end)  

    string s1 = "Hello";
    string s2 = "World";
    string s3 = s1 + " " + s2; // "Hello World"

    string line;
    cout << "Enter a line: ";
    getline(cin, line);
    cout << "You entered: " << line << endl;

    //replace all
    string part;
    stringstream ss("one,two,three");
    while (getline(ss, part, ',')) {
        cout << "Part: " << part << endl;
    }

    string a1 = "a b c d e a a b b c";
    int pos = 0;
    string newStr = "r";
    string oldStr = "a";
    while((pos = a1.find(newStr, pos)) != string::npos)
    {
        a1.replace(pos, oldStr.length(), newStr);
        pos += newStr.length();
    }

    //transform lower
    transform(a1.begin(), a1.end(), a1.begin(), ::tolower);
    transform(a1.begin(), a1.end(), a1.begin(), ::toupper);

    vector<int> ascii_values(a1.size());

    // Transform each character to its integer ASCII value
    transform(text.begin(), text.end(), ascii_values.begin(),
                   [](char c) { return static_cast<int>(c); });

    char censor_vowels(char c) {
        c = tolower(c);
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
            return '*';
        }
        return c;
    }
                   
    transform(a1.begin(), a1.end(), a1.begin(), censor_vowels);

    transform(a1.begin(), a1.end(), a1.begin(),
        [](char c) { return static_cast<int>(c);} );

    //unique
    a1.erase(unique(a1.begin(), a1.end()), a1.end());

    //parsing
    string input = "Hello 123 World";
    stringstream ss(input);

    string token;
    vector<string> save;
    
    while (ss >> token)
    {
        save.push_back(token);
    }

    string numString;
    int num = 0;
    for (const auto& c: input)
    {
        if(isdigit(c))
        {
            numString += c;
        }
        else{
            if(!numString.empty())
            {
                num = stoi(numString);
                numString.clear();
            }
        }
    }

    string input = "42 A Hello World";
    stringstream ss(input);
    int num;
    char character;
    string str;

    ss >> num >> character >> str;


    string input;
    //input string with space, whole line
    getline(cin, input);

    stringstream ss(input);

    vector<string> arr;
    string token;
    while(ss >> token){
        arr.push_back(token);
    }

    return 0;
}