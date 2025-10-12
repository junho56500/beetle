#include <iostream>

using namespace std;

// 1. Abstract Products
class Device {
public:
    virtual ~Device() = default;
    virtual void compute() const = 0;
};

//2. Concrete Products
// Concrete Product A
class CPUDevice : public Device {
public:
    void compute() const override {
        std::cout << "Computing on CPU.\n";
    }
};

// Concrete Product B
class GPUDevice : public Device {
public:
    void compute() const override {
        std::cout << "Computing on GPU.\n";
    }
};

//3. Abstract Factory
// Abstract Creator with Factory Method
class DeviceFactory {
public:
    // The Factory Method - returns the Abstract Product type
    virtual Device* createDevice() const = 0;
    
    virtual ~DeviceFactory() = default;

    // Core logic that uses the product, independent of concrete product type
    void runDevice() const {
        // The factory method is called to create a product.
        Device* device = this->createDevice();
        
        // The client logic works with the abstract product interface.
        device->compute();

        delete device;
    }
};

//4. Concrete Factories
// Concrete Factory for CPU
class CPUDeviceFactory : public DeviceFactory {
public:
    Device* createDevice() const override {
        // Returns the Concrete Product
        return new CPUDevice();
    }
};

// Concrete Factory for GPU
class GPUDeviceFactory : public DeviceFactory {
public:
    Device* createDevice() const override {
        // Returns the Concrete Product
        return new GPUDevice();
    }
};

// Main function demonstration
int main() {
    std::cout << "Client: Launching with CPU Factory:\n";
    DeviceFactory* cpuFactory = new CPUDeviceFactory();
    cpuFactory->runDevice();

    std::cout << "\nClient: Launching with GPU Factory:\n";
    DeviceFactory* gpuFactory = new GPUDeviceFactory();
    gpuFactory->runDevice();

    delete cpuFactory;
    delete gpuFactory;

    return 0;
}