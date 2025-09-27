#include <iostream>
#include <string>

int main() {
    std::string text = "C++ strings are great for coding!";
    std::string substring_to_find = "great";

    // 1. Find the first occurrence of the substring
    size_t found_pos = text.find(substring_to_find);

    
    // 2. Check if the substring was found
    if (found_pos != std::string::npos) {
        std::cout << "The substring was found at position: " << found_pos << std::endl;
        
        // You can use the returned position to get the substring
        std::cout << "The found part is: \"" << text.substr(found_pos) << "\"" << std::endl;

    } else {
        std::cout << "The substring was not found." << std::endl;
    }

    // 3. Find another occurrence starting from a specific position
    std::string text_repeat = "The cat in the hat.";
    size_t second_cat_pos = text_repeat.find("cat", 4); // Start searching from index 4

    if (second_cat_pos != std::string::npos) {
        std::cout << "The second 'cat' was found at position: " << second_cat_pos << std::endl;
    } else {
        std::cout << "The second 'cat' was not found." << std::endl;
    }

    return 0;
}