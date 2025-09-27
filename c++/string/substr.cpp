#include <iostream>
#include <string>

using namespace std;
// Function to simulate a trim operation
// Trims leading and trailing whitespace characters
string trim(const string& str) {
    const string whitespace = " \t\n\r\f\v";
    size_t start = str.find_first_not_of(whitespace);
    if (string::npos == start) {
        return "";
    }
    size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

int main() {
    string my_string = "   Hello, world!   ";
    string search_string = "world";
    string another_string = "Hello, world!";

    // 1. Simulating indexOf with string::find
    size_t index = my_string.find(search_string);
    if (index != string::npos) {
        cout << "1. 'world' found at index: " << index << endl;
    } else {
        cout << "1. 'world' not found." << endl;
    }

    // 2. Substring with string::substr
    // Note: substr() takes a start index and length
    string sub = my_string.substr(3, 5);
    cout << "2. Substring from index 3 with length 5: '" << sub << "'" << endl;

    // 3. Replace
    string replaced_string = my_string;
    // The replace method takes a start position, length, and the new string
    replaced_string.replace(my_string.find("world"), string("world").length(), "C++");
    cout << "3. Replaced string: '" << replaced_string << "'" << endl;

    // 4. Trim with a custom function
    string trimmed_string = trim(my_string);
    cout << "4. Trimmed string: '" << trimmed_string << "'" << endl;

    // 5. CompareTo with string::compare
    // Returns 0 if equal, < 0 if less than, > 0 if greater than
    int comparison_result = my_string.compare(another_string);
    if (comparison_result == 0) {
        cout << "5. Strings are equal." << endl;
    } else {
        cout << "5. Strings are not equal. Result: " << comparison_result << endl;
    }

    // A more common way to compare for equality is with the == operator
    if (my_string == another_string) {
        cout << "5. Strings are equal (using ==)." << endl;
    } else {
        cout << "5. Strings are not equal (using ==)." << endl;
    }

    return 0;
}