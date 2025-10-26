#include <iostream>
#include <vector>
#include <string>

// --- Data Structure ---
struct Configuration {
    std::string name;
    int max_threads = 4;
};

// --- Modern C++ (Non-Null Guaranteed) Approach ---
// By taking 'config' as a reference (&), the caller MUST provide a valid, 
// initialized object. This guarantees 'config' is NOT null.
void process_configuration_safe(Configuration& config) {
    // LESSON 1: No need to check for null
    std::cout << "[SAFE] Processing configuration: " << config.name 
              << " with max threads: " << config.max_threads << std::endl;
    
    // We can safely modify the object pointed to by the reference
    config.max_threads *= 2;
}

// --- Traditional C++ (Nullable) Approach for Comparison ---
// By taking 'config_ptr' as a raw pointer (*), the caller MAY pass nullptr.
void process_configuration_unsafe(Configuration* config_ptr) {
    // LESSON 2: MUST check for null to prevent crashes
    if (config_ptr == nullptr) {
        std::cerr << "[UNSAFE] Error: Received a null pointer. Aborting process." << std::endl;
        return;
    }
    
    std::cout << "[UNSAFE] Processing configuration: " << config_ptr->name 
              << " with max threads: " << config_ptr->max_threads << std::endl;
}

int main() {
    // 1. Valid, Initialized Object
    Configuration main_config = {"SystemA", 8};
    
    std::cout << "--- Calling Safe Function (References) ---" << std::endl;
    
    // Passing the valid object by reference is clean and guaranteed safe
    process_configuration_safe(main_config);
    // The change is visible
    std::cout << "Main config updated max threads to: " << main_config.max_threads << std::endl;

    
    // --- Demonstrating Non-Null Enforcement ---
    // If you try to pass an uninitialized reference, the compiler will error:
    // Configuration& bad_config; // ERROR: Reference must be initialized
    
    // If you try to call 'process_configuration_safe' with a null state, 
    // you must dereference a null pointer *before* the call, causing a crash 
    // or compilation error, preventing the bad state from entering the function.
    // Example (requires raw pointer):
    Configuration* null_ptr = nullptr;
    // process_configuration_safe(*null_ptr); // CRASH! (But the function signature didn't allow 'nullptr')
    
    std::cout << "\n--- Calling Unsafe Function (Raw Pointers) ---" << std::endl;

    // Passing a valid pointer (Safe call)
    process_configuration_unsafe(&main_config);
    
    // Passing a null pointer (Unsafe call - requires runtime check)
    process_configuration_unsafe(null_ptr);

    // LESSON 3: Using references clearly communicates intent: 
    // "This object must exist and I will not check if it's null."
    
    return 0;
}