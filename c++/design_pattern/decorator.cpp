
// 1. Component Interface (Beverage.h)
// This is the base interface for both the basic coffee and all its decorators.
// Beverage.h
#include <iostream>
#include <string>

class Beverage {
public:
    virtual ~Beverage() = default;
    virtual std::string getDescription() const = 0;
    virtual double getCost() const = 0;
};

// 2. Concrete Component (Espresso.h)
// The basic object we'll be decorating.
// Espresso.h
#include "Beverage.h"

class Espresso : public Beverage {
public:
    std::string getDescription() const override {
        return "Espresso";
    }
    double getCost() const override {
        return 1.99;
    }
};

// 3. Decorator Base Class (CondimentDecorator.h)
// This abstract decorator maintains a pointer to a Beverage and ensures all decorators conform to the Beverage interface.
// CondimentDecorator.h
#include "Beverage.h"

class CondimentDecorator : public Beverage {
protected:
    // This is the core of the pattern: a reference to the wrapped object
    Beverage* beverage; 

public:
    CondimentDecorator(Beverage* b) : beverage(b) {}
    ~CondimentDecorator() override { 
        // Important: ensure the decorator does not delete the component if it doesn't own it
        // For simplicity here, we assume ownership for this example's structure.
        delete beverage; 
    } 

    // All methods must be implemented or delegated
    std::string getDescription() const override { 
        return beverage->getDescription(); // Delegate to the wrapped component
    }
    double getCost() const override {
        return beverage->getCost(); // Delegate to the wrapped component
    }
};

// 4. Concrete Decorators (Milk.h, Whip.h)
// These classes add specific functionalities (cost and description) to the beverage they wrap.
// Milk.h
#include "CondimentDecorator.h"

class Milk : public CondimentDecorator {
public:
    Milk(Beverage* b) : CondimentDecorator(b) {}

    std::string getDescription() const override {
        // Adds its description to the wrapped component's description
        return beverage->getDescription() + ", Milk";
    }
    double getCost() const override {
        // Adds its cost to the wrapped component's cost
        return beverage->getCost() + 0.50;
    }
};

// Whip.h
class Whip : public CondimentDecorator {
public:
    Whip(Beverage* b) : CondimentDecorator(b) {}

    std::string getDescription() const override {
        return beverage->getDescription() + ", Whip";
    }
    double getCost() const override {
        return beverage->getCost() + 0.75;
    }
};

// Client Code (main.cpp)
// The client can dynamically compose objects by wrapping them in multiple decorators.
// main.cpp
#include "Espresso.h"
#include "Milk.h"
#include "Whip.h"

void printOrder(const Beverage* beverage) {
    std::cout << "Order: " << beverage->getDescription() 
              << " | Cost: $" << beverage->getCost() << "\n";
}

int main() {
    // 1. Order a simple Espresso
    Beverage* coffee1 = new Espresso();
    printOrder(coffee1);
    // Output: Order: Espresso | Cost: $1.99

    std::cout << "\n";

    // 2. Order an Espresso with Milk and Whip (chaining decorators)
    // Start with the basic component (Espresso)
    Beverage* coffee2 = new Espresso();
    
    // Wrap it in Milk
    coffee2 = new Milk(coffee2); 
    
    // Wrap the Milk-wrapped object in Whip
    coffee2 = new Whip(coffee2); 

    printOrder(coffee2);
    // Output: Order: Espresso, Milk, Whip | Cost: $3.24 (1.99 + 0.50 + 0.75)

    // Note: The cleanup must happen on the outermost decorator
    delete coffee1;
    delete coffee2; // The Whip destructor will call the Milk destructor, 
                    // which calls the Espresso destructor, cleaning up the whole stack.

    return 0;
}
