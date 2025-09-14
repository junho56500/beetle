#include <iostream>
#include <string>
#include <vector>
#include <memory>

// Base class
class Animal {
public:
    virtual void makeSound() = 0; // Pure virtual function
    virtual ~Animal() = default;
};

// Derived class 1
class Dog : public Animal {
public:
    void makeSound() override {
        std::cout << "Woof!" << std::endl;
    }
    void fetchBall() {
        std::cout << "Fetching the ball." << std::endl;
    }
};

// Derived class 2
class Cat : public Animal {
public:
    void makeSound() override {
        std::cout << "Meow!" << std::endl;
    }
    void scratchFurniture() {
        std::cout << "Scratching the furniture." << std::endl;
    }
};

void checkAnimalType(Animal* animal) {
    // Try to safely cast the base class pointer to a Dog pointer
    Dog* dog = dynamic_cast<Dog*>(animal);
    
    // Check if the cast was successful (returns nullptr on failure)
    if (dog != nullptr) {
        std::cout << "This is a dog. ";
        dog->fetchBall(); // Now it's safe to call the derived class method
    } else {
        // Try to safely cast the base class pointer to a Cat pointer
        Cat* cat = dynamic_cast<Cat*>(animal);
        if (cat != nullptr) {
            std::cout << "This is a cat. ";
            cat->scratchFurniture(); // Now it's safe to call the derived class method
        } else {
            std::cout << "This is an unknown animal type." << std::endl;
        }
    }
}


// using namespace std;

// class animal{
// public:
//     virtual void checkType() = 0;
//     virtual ~animal() = default;
// };

// class cat : public animal{
// public:
//     void checkType() override{
//         cout << "cat" << endl;
//     }
//     void rollThread() {
//         cout << "rollThread" << endl;
//     }
// };

// class dog : public animal{
// public:
//     void checkType() override{
//         cout << "dog" << endl;
//     }
//     void bark(){
//         cout << "bark!!" << endl;
//     }
// };

// void checkAnimal(shared_ptr<animal> a)
// {
//     shared_ptr<cat> c = dynamic_pointer_cast<cat>(a);

//     if(a == nullptr)
//     {
//         shared_ptr<dog> d = dynamic_pointer_cast<dog>(a);
//         d->bark();
//     }
//     else
//     {
//         c->rollThread();
//     }
// }

// int main()
// {
//     shared_ptr<cat> cat1 = make_shared<cat>();
//     shared_ptr<dog> dog1 = make_shared<dog>();

//     checkAnimal(cat1);
//     checkAnimal(dog1);

// }



// --- For static_cast and dynamic_cast examples ---
class Base {
public:
    virtual void show() { std::cout << "Base class." << std::endl; }
};

class Derived : public Base {
public:
    void show() override { std::cout << "Derived class." << std::endl; }
    void specializedMethod() { std::cout << "Specialized method from Derived." << std::endl; }
};

int main() {
    Dog myDog;
    Cat myCat;
    
    // Pass a Dog object to the function
    std::cout << "Checking myDog:" << std::endl;
    checkAnimalType(&myDog);
    
    // Pass a Cat object to the function
    std::cout << "\nChecking myCat:" << std::endl;
    checkAnimalType(&myCat);
    
    // --- 1. static_cast ---
    std::cout << "--- static_cast ---" << std::endl;
    int a = 10;
    double b = static_cast<double>(a); // Conversion of an int to a double
    std::cout << "int to double: " << b << std::endl;
    // Down-casting (less safe, but useful for related types)
    Derived d_obj;
    Base* base_ptr = &d_obj; // Up-casting is safe
    Derived* derived_ptr = static_cast<Derived*>(base_ptr);
    derived_ptr->show();
    
    // --- 2. dynamic_cast ---
    std::cout << "\n--- dynamic_cast ---" << std::endl;
    Base* base_ptr_1 = new Derived;
    Base* base_ptr_2 = new Base;
    // Safe down-casting
    Derived* derived_ptr_safe = dynamic_cast<Derived*>(base_ptr_1);
    if (derived_ptr_safe) {
        derived_ptr_safe->specializedMethod();
    } else {
        std::cout << "Cast failed for base_ptr_1." << std::endl;
    }
    // Failed cast
    Derived* derived_ptr_unsafe = dynamic_cast<Derived*>(base_ptr_2);
    if (derived_ptr_unsafe) {
        derived_ptr_unsafe->specializedMethod();
    } else {
        std::cout << "Cast failed for base_ptr_2. Returns nullptr." << std::endl;
    }
    
    // --- 3. const_cast ---
    std::cout << "\n--- const_cast ---" << std::endl;
    const int const_value = 100;
    int* ptr = const_cast<int>(const_value); // Removing constness
    *ptr = 200; // Undefined behavior, but the cast is successful
    std::cout << "Value after const_cast: " << *ptr << std::endl;
    
    // --- 4. reinterpret_cast ---
    std::cout << "\n--- reinterpret_cast ---" << std::endl;
    long address = reinterpret_cast<long>(ptr); // Casting pointer to integer type
    std::cout << "Pointer as integer: " << address << std::endl;
    
    delete base_ptr_1;
    delete base_ptr_2;

    return 0;
}