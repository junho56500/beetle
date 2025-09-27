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
    std::unique_ptr<MyClass> ptr1 = std::make_unique<MyClass>();
    ptr1->greet();

    // Ownership can be transferred using std::move:
    std::unique_ptr<MyClass> ptr2 = std::move(ptr1);
    if (!ptr1) {
        std::cout << "ptr is now null\n";
    }


    std::shared_ptr<MyClass> ptr3 = std::make_shared<MyClass>();
    {
        std::shared_ptr<MyClass> ptr4 = ptr3; // Reference count increases
        ptr4->greet();
        std::cout << "ptr4 goes out of scope\n";
    } // ptr4 destroyed, but ptr1 still owns the object
    std::cout << "ptr3 still owns the object\n";
    
}