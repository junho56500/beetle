#include "stdio.h"
#include <iostream>
#include <memory>

using namespace std;

class MyClass {
    public:
        MyClass() { std::cout << "Constructed\n"; }
        ~MyClass() { std::cout << "Destroyed\n"; }
        void greet() { std::cout << "Hello from MyClass\n"; }
    };

int main() {
    std::unique_ptr<MyClass> ptr = std::make_unique<MyClass>();
    ptr->greet();

    // Ownership can be transferred using std::move:
    std::unique_ptr<MyClass> ptr2 = std::move(ptr);
    if (!ptr) {
        std::cout << "ptr is now null\n";
    }
    
}