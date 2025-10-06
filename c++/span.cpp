#include <iostream>
#include <vector>
#include <numeric> // For std::accumulate
#include <span>    // Standard header for std::span (C++20)

// 1. Inefficient: Passes by value (creates a COPY of the entire vector)
// This should almost always be avoided for large containers.
void process_data_copy(std::vector<int> data) {
    std::cout << "\n[1] Passed by Value (std::vector<int> data):";
    std::cout << "\n   -> A full vector copy is created (HIGH OVERHEAD).";
    std::cout << "\n   -> Sum: " << std::accumulate(data.begin(), data.end(), 0);
}

// 2. Standard: Passes by const reference (efficient, zero copy)
// This is efficient but is tightly coupled to accepting only 'std::vector'.
// NOTE: 'const' applies to the entire vector object, preventing modification of both
// the vector's size and its elements. (e.g., data.push_back() is illegal).
void process_data_ref(const std::vector<int>& data) {
    std::cout << "\n[2] Passed by Const Reference (const std::vector<int>& data):";
    std::cout << "\n   -> No copy, but the function ONLY accepts std::vector.";
    std::cout << "\n   -> Sum: " << std::accumulate(data.begin(), data.end(), 0);
}

// 3. Modern C++20: Passes by Span (Efficient, flexible, zero copy)
// std::span provides a non-owning, zero-overhead view over any contiguous range.
void process_data_span(std::span<const int> data) {
    // Note the use of 'const int'. The 'const' here applies only to the elements
    // viewed by the span, ensuring the function cannot write to the underlying data.
    std::cout << "\n[3] Passed by std::span<const int>:";
    std::cout << "\n   -> Non-owning view. Works with vector, C-arrays, and array-like objects.";
    std::cout << "\n   -> Sum: " << std::accumulate(data.begin(), data.end(), 0);
}

int main() {
    std::cout << "C++ Function Parameter Comparison: vector vs span (C++20)\n" << std::endl;

    // --- Data Sources ---
    std::vector<int> my_vector = {10, 20, 30, 40, 50};
    int c_array[] = {1, 2, 3, 4, 5};
    
    std::cout << "--- Calling with std::vector ---";

    // Call 1: Copy is created (Inefficient)
    process_data_copy(my_vector);
    
    // Call 2: Reference is passed (Efficient, but rigid)
    process_data_ref(my_vector);
    
    // Call 3: Span view is passed (Efficient and Flexible)
    process_data_span(my_vector); 
    
    std::cout << "\n\n--- Calling with C-style Array ---";

    // Only the span-based function can easily accept a C-style array without manual conversion
    // (A key flexibility advantage of span)
    process_data_span(c_array);

    return 0;
}
