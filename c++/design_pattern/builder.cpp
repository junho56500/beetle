// 1. Product (Computer.h)
// The complex object we want to build.
// Computer.h
#include <iostream>
#include <string>

class Computer {
private:
    std::string cpu;
    std::string motherboard;
    std::string ram;

public:
    void setCPU(const std::string& c) { cpu = c; }
    void setMotherboard(const std::string& m) { motherboard = m; }
    void setRAM(const std::string& r) { ram = r; }

    void showConfiguration() const {
        std::cout << "--- Computer Configuration ---\n";
        std::cout << "CPU: " << cpu << "\n";
        std::cout << "Motherboard: " << motherboard << "\n";
        std::cout << "RAM: " << ram << "\n";
        std::cout << "----------------------------\n";
    }
};

// 2. Builder Interface (ComputerBuilder.h)
// The interface defines the necessary steps for building a computer.
// ComputerBuilder.h
#include "Computer.h"

class ComputerBuilder {
public:
    virtual ~ComputerBuilder() = default;

    // Building steps
    virtual void buildCPU() = 0;
    virtual void buildMotherboard() = 0;
    virtual void buildRAM() = 0;

    // Method to retrieve the final product
    virtual Computer* getResult() = 0;
};


// 3. Concrete Builder (GamingPCBuilder.h)
// This builder implements the steps to create a specific configuration (a high-performance Gaming PC).
// GamingPCBuilder.h
class GamingPCBuilder : public ComputerBuilder {
private:
    Computer* computer;

public:
    GamingPCBuilder() { computer = new Computer(); }
    ~GamingPCBuilder() override { delete computer; }

    void buildCPU() override {
        computer->setCPU("Intel Core i9 (High-End)");
    }

    void buildMotherboard() override {
        computer->setMotherboard("Z690 Gaming Edition");
    }

    void buildRAM() override {
        computer->setRAM("32GB DDR5 6000MHz");
    }

    Computer* getResult() override {
        // Return the product and detach it from the builder
        Computer* finalProduct = computer;
        computer = new Computer(); // Reset for a new build
        return finalProduct;
    }
};

// 4. Director (AssemblyDirector.h)
// The Director knows the general sequence for assembly, but it doesn't know the specifics of the parts. It works with any ComputerBuilder.
// AssemblyDirector.h
#include "ComputerBuilder.h"

class AssemblyDirector {
public:
    void construct(ComputerBuilder* builder) {
        builder->buildMotherboard(); // Step 1: Install Motherboard first
        builder->buildCPU();         // Step 2: Install CPU
        builder->buildRAM();         // Step 3: Install RAM
    }
};

// Client Code (main.cpp)
// The client uses the Director and a specific Builder to get the desired Product.

// main.cpp
#include "AssemblyDirector.h"
#include "GamingPCBuilder.h"

int main() {
    AssemblyDirector director;
    GamingPCBuilder builder;

    // 1. Tell the Director to execute the standard construction sequence
    director.construct(&builder);

    // 2. Get the fully assembled Product from the Builder
    Computer* gamingPC = builder.getResult();

    // 3. Use the product
    gamingPC->showConfiguration();

    // Output:
    // --- Computer Configuration ---
    // CPU: Intel Core i9 (High-End)
    // Motherboard: Z690 Gaming Edition
    // RAM: 32GB DDR5 6000MHz
    // ----------------------------

    // The client cleans up the final product
    delete gamingPC;

    // If another builder (e.g., OfficePCBuilder) was used, 
    // the Director and client code would remain unchanged.

    return 0;
}