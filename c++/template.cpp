#include <iostream>
#include <string>
#include <array>
#include <algorithm> // For std::transform
#include <cctype>    // For ::toupper

#include <vector>
#include <type_traits> // For std::is_arithmetic
#include <variant>

// ## 1. Primary Class Template (The General Rule)
// Takes a type 'T' and a non-type 'size_t Size' as parameters.
template <typename T, size_t Size>
class StaticArray {
private:
    std::array<T, Size> m_data;

public:
    // Constructor to initialize the array
    StaticArray(const std::array<T, Size>& items) : m_data(items) {}

    // A generic transformation: multiply each element by 2.
    // This works for any numeric type.
    void transformAndPrint() const {
        std::cout << "Generic transformation (doubling):" << std::endl;
        for (const auto& item : m_data) {
            std::cout << item * 2 << " ";
        }
        std::cout << std::endl;
    }
};

// ## 2. Template Specialization for std::string (The Exception)
// This is a special version of the template that the compiler will use
// ONLY when T is std::string. The size 'Size' is still generic.
template <size_t Size>
class StaticArray<std::string, Size> {
private:
    std::array<std::string, Size> m_data;

public:
    StaticArray(const std::array<std::string, Size>& items) : m_data(items) {}

    // A specific transformation for strings: convert to uppercase.
    void transformAndPrint() const {
        std::cout << "Specialized transformation (uppercase):" << std::endl;
        for (const auto& item : m_data) {
            std::string upper_item;
            // Reserve memory to be efficient
            upper_item.reserve(item.length());
            // Transform the string to uppercase and print
            std::transform(item.begin(), item.end(), std::back_inserter(upper_item), ::toupper);
            std::cout << upper_item << " ";
        }
        std::cout << std::endl;
    }
};


template <typename T>
void printDetails(const T& value) {
    std::cout << "Data: " << value << " | ";

    // This block is evaluated at COMPILE TIME
    if constexpr (std::is_arithmetic_v<T>) {
        // This code only exists in instantiations for numbers (int, float, double, etc.)
        std::cout << "Type: It's a number." << std::endl;
    } else if constexpr (std::is_same_v<T, std::string>) {
        // This code only exists in instantiations for std::string
        std::cout << "Type: It's a string with length " << value.length() << "." << std::endl;
    } else {
        // This is the fallback for any other type
        std::cout << "Type: It's some other type." << std::endl;
    }
}


int main() {
    // Example 1: Using the primary template with 'int' and size 5.
    // The compiler generates the generic version of the class.
    std::array<int, 5> numbers = {1, 2, 3, 4, 5};
    StaticArray<int, 5> intArray(numbers);
    intArray.transformAndPrint(); // Output will be doubled numbers.

    std::cout << "\n--------------------\n\n";

    // Example 2: Using the specialized template with 'std::string' and size 3.
    // The compiler sees T=std::string and uses our specialized version.
    std::array<std::string, 3> words = {"hello", "world", "!"};
    StaticArray<std::string, 3> stringArray(words);
    stringArray.transformAndPrint(); // Output will be uppercase strings.

    return 0;
}

// Yes, modern C++ provides several powerful and type-safe ways to execute different code for different data types at compile time. The most direct and common method since C++17 is if constexpr.

// if constexpr allows the compiler to evaluate a condition at compile time and discard the code branches that don't apply. This means you can write code inside a template that would only be valid for certain types, without causing a compilation error for other types.

// Here is an example of a generic printDetails function that behaves differently for numbers, strings, and other types.



// int main() {
//     printDetails(101);                  // Uses the std::is_arithmetic_v branch
//     printDetails(3.14);                 // Also uses the std::is_arithmetic_v branch
//     printDetails(std::string("hello")); // Uses the std::is_same_v branch
//     printDetails("C-style literal");    // Uses the 'else' branch (type is const char*)
    
//     return 0;
// }