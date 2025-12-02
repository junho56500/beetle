#include <iostream>
#include <type_traits> // Required for std::enable_if_t and std::is_same

// --- Template 1: Constrained for 'int' only ---
template<
    typename T, 
    // Use SFINAE: Only allow substitution if T is exactly 'int'.
    std::enable_if_t<std::is_same_v<T, int>, bool> = true 
>
void process_number(T value) {
    std::cout << "[INT Handler] Processing integer value: " << value 
              << ". Applying integer-specific logic (e.g., bitwise ops)." << std::endl;
}

// --- Template 2: Constrained for 'float' only ---
template<
    typename T, 
    // Use SFINAE: Only allow substitution if T is exactly 'float'.
    std::enable_if_t<std::is_same_v<T, float>, bool> = true
>
void process_number(T value) {
    // Note: The template parameter name 'T' is the same, but the constraints make them distinct functions.
    std::cout << "[FLOAT Handler] Processing float value: " << value 
              << ". Applying float-specific logic (e.g., precision rounding)." << std::endl;
}

// --- Main execution ---
int main() {
    std::cout << "--- Demonstrating SFINAE-based Overloading ---" << std::endl;

    int i = 42;
    float f = 3.14159f;
    double d = 2.71828;
    
    // 1. Calls the 'int' constrained template (Template 1)
    process_number(i); 

    // 2. Calls the 'float' constrained template (Template 2)
    process_number(f); 

    std::cout << "\n--- Testing a non-matching type (double) ---" << std::endl;
    // 3. Error Case: double does not match 'int' or 'float'.
    // The compiler attempts substitution for both templates, both fail, 
    // and since no unconstrained function exists, compilation fails here.
    
    // If you uncomment the line below, it will cause a compile-time error:
    // process_number(d); 

    std::cout << "Attempting to call 'process_number(double)' would result in a "
              << "compile-time error because neither template is enabled for 'double'." << std::endl;

    return 0;
}
