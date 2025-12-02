#include <iostream>

using namespace std;

// 1. Abstract Products
// Abstract Product A
class Button {
public:
    virtual ~Button() = default;
    virtual void render() const = 0;
};

// Abstract Product B
class Checkbox {
public:
    virtual ~Checkbox() = default;
    virtual void paint() const = 0;
};

//2. Concrete Products
class WindowsButton : public Button {
public:
    void render() const override {
        std::cout << "Rendering a Windows Button.\n";
    }
};

class WindowsCheckbox : public Checkbox {
public:
    void paint() const override {
        std::cout << "Painting a Windows Checkbox.\n";
    }
};


class MacButton : public Button {
public:
    void render() const override {
        std::cout << "Rendering a macOS Button.\n";
    }
};

class MacCheckbox : public Checkbox {
public:
    void paint() const override {
        std::cout << "Painting a macOS Checkbox.\n";
    }
};

//3. Abstract Factory
class GUIFactory {
public:
    virtual ~GUIFactory() = default;
    virtual Button* createButton() const = 0;
    virtual Checkbox* createCheckbox() const = 0;
};

//4. Concrete Factories
class WindowsFactory : public GUIFactory {
public:
    Button* createButton() const override {
        return new WindowsButton();
    }
    Checkbox* createCheckbox() const override {
        return new WindowsCheckbox();
    }
};

class MacFactory : public GUIFactory {
public:
    Button* createButton() const override {
        return new MacButton();
    }
    Checkbox* createCheckbox() const override {
        return new MacCheckbox();
    }
};

//5. Client Code (Usage)
void client_code(const GUIFactory* factory) {
    const Button* button = factory->createButton();
    const Checkbox* checkbox = factory->createCheckbox();

    button->render();
    checkbox->paint();

    delete button;
    delete checkbox;
}

// Main function demonstration
int main() {
    std::cout << "Client: Testing Windows family components.\n";
    GUIFactory* windowsFactory = new WindowsFactory();
    client_code(windowsFactory);
    delete windowsFactory;

    std::cout << "\nClient: Testing macOS family components.\n";
    GUIFactory* macFactory = new MacFactory();
    client_code(macFactory);
    delete macFactory;

    return 0;
}