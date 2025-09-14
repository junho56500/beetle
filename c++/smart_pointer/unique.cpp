#include <iostream>
#include <memory>
#include <string>

// A simple class to demonstrate smart pointer lifetime management.
// The constructor and destructor will print messages to the console.
class MyObject {
public:
    std::string name;

    MyObject(const std::string& objectName) : name(objectName) {
        std::cout << "  - MyObject '" << name << "' created." << std::endl;
    }

    ~MyObject() {
        std::cout << "  - MyObject '" << name << "' destroyed." << std::endl;
    }

    void doSomething() {
        std::cout << "  - MyObject '" << name << "' is doing something." << std::endl;
    }
};

// --- Part 1: std::unique_ptr ---
// Demonstrates exclusive, non-shared ownership.
void demonstrateUniquePtr() {
    std::cout << "--- Demonstrating std::unique_ptr ---" << std::endl;

    // Use std::make_unique for exception-safe and efficient creation.
    // The unique_ptr 'ptr1' now has exclusive ownership of the object.
    std::unique_ptr<MyObject> ptr1 = std::make_unique<MyObject>("UniqueObject");
    ptr1->doSomething();

    // The following line would cause a compile-time error because
    // std::unique_ptr cannot be copied. Its copy constructor is deleted.
    // std::unique_ptr<MyObject> ptr2 = ptr1; // ERROR!

    // Ownership can be transferred using std::move. After the move,
    // ptr1 will be empty (nullptr).
    std::unique_ptr<MyObject> ptr2 = std::move(ptr1);

    if (ptr1 == nullptr) {
        std::cout << "  - Ownership transferred. ptr1 is now empty." << std::endl;
    }

    // Now ptr2 has exclusive ownership.
    if (ptr2 != nullptr) {
        ptr2->doSomething();
    }
    
    // The object will be automatically destroyed when ptr2 goes out of scope.
    std::cout << "--- std::unique_ptr demonstration finished ---" << std::endl;
}

// --- Part 2: std::shared_ptr ---
// Demonstrates shared ownership using a reference count.
void demonstrateSharedPtr() {
    std::cout << "\n--- Demonstrating std::shared_ptr ---" << std::endl;

    // Use std::make_shared for a single, efficient allocation.
    std::shared_ptr<MyObject> shared_ptr1 = std::make_shared<MyObject>("SharedObject");

    // Check the initial reference count.
    std::cout << "  - Reference count for 'shared_ptr1': " << shared_ptr1.use_count() << std::endl;

    // Create more shared pointers that share ownership.
    std::shared_ptr<MyObject> shared_ptr2 = shared_ptr1;
    std::shared_ptr<MyObject> shared_ptr3 = shared_ptr1;

    // The reference count increases with each new owner.
    std::cout << "  - Reference count after two copies: " << shared_ptr1.use_count() << std::endl;

    // Resetting one pointer decreases the count.
    shared_ptr2.reset();
    std::cout << "  - Reference count after shared_ptr2 is reset: " << shared_ptr1.use_count() << std::endl;

    // The object is only destroyed when the last shared_ptr is destroyed.
    // This will happen at the end of this function's scope.
    std::cout << "--- std::shared_ptr demonstration finished ---" << std::endl;
}

// A simple node class to demonstrate circular references.
class Node {
public:
    std::shared_ptr<Node> next;
    std::string name;

    Node(const std::string& nodeName) : name(nodeName) {
        std::cout << "  - Node '" << name << "' created." << std::endl;
    }
    ~Node() {
        std::cout << "  - Node '" << name << "' destroyed." << std::endl;
    }
};

// --- Part 3: std::weak_ptr ---
// Demonstrates how to observe an object without affecting its lifetime.
// It is crucial for breaking circular references.
void demonstrateWeakPtr() {
    std::cout << "\n--- Demonstrating std::weak_ptr ---" << std::endl;

    std::shared_ptr<MyObject> shared_ptr = std::make_shared<MyObject>("WeakPtrObject");
    // Create a weak_ptr from the shared_ptr. The reference count does not increase.
    std::weak_ptr<MyObject> weak_ptr = shared_ptr;
    
    std::cout << "  - shared_ptr reference count: " << shared_ptr.use_count() << std::endl;

    // Check if the object still exists.
    if (auto sp = weak_ptr.lock()) {
        std::cout << "  - weak_ptr is not expired. Object still exists." << std::endl;
        sp->doSomething();
    }

    // Reset the shared_ptr. The object is now destroyed.
    shared_ptr.reset();
    
    std::cout << "  - shared_ptr is reset. Object should be destroyed." << std::endl;

    // Now, the weak_ptr is expired.
    if (weak_ptr.expired()) {
        std::cout << "  - weak_ptr is now expired. Object has been destroyed." << std::endl;
    }

    std::cout << "\n--- Demonstrating weak_ptr to break circular references ---" << std::endl;
    // Circular reference example with shared_ptr. This would leak memory.
    std::cout << "--- Shared pointers will leak memory in this example ---" << std::endl;
    {
        std::shared_ptr<Node> node1 = std::make_shared<Node>("Node 1");
        std::shared_ptr<Node> node2 = std::make_shared<Node>("Node 2");
        node1->next = node2;
        node2->next = node1;
        // At the end of this scope, node1 and node2 will not be destroyed
        // because their shared_ptr reference count never reaches zero.
    }
    std::cout << "  - Check console output. 'Node 1' and 'Node 2' were not destroyed. Memory leaked." << std::endl;

    std::cout << "\n--- Using weak_ptr to prevent circular references ---" << std::endl;
    // To fix this, one of the pointers should be a weak_ptr.
    class FixedNode {
    public:
        std::shared_ptr<FixedNode> next;
        std::weak_ptr<FixedNode> prev; // Use weak_ptr to avoid cycle
        std::string name;

        FixedNode(const std::string& nodeName) : name(nodeName) {
            std::cout << "  - FixedNode '" << name << "' created." << std::endl;
        }
        ~FixedNode() {
            std::cout << "  - FixedNode '" << name << "' destroyed." << std::endl;
        }
    };

    {
        std::shared_ptr<FixedNode> node1 = std::make_shared<FixedNode>("FixedNode 1");
        std::shared_ptr<FixedNode> node2 = std::make_shared<FixedNode>("FixedNode 2");
        node1->next = node2;
        node2->prev = node1; // Assigning weak_ptr does not increase reference count
    }
    std::cout << "  - Check console output. Both nodes were properly destroyed." << std::endl;
    
    std::cout << "--- std::weak_ptr demonstration finished ---" << std::endl;
}

int main() {
    demonstrateUniquePtr();
    demonstrateSharedPtr();
    demonstrateWeakPtr();
    
    return 0;
}
