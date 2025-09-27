#include <iostream>

struct Node {
    int data;
    Node* next;

    Node(int val):data(val), next(nullptr) {}
};

class LinkedList {
private:
    Node* head;

public:
    // Constructor to initialize an empty list
    LinkedList() : head(nullptr) {}

    // Function to add a new node to the end of the list
    void append(int val) {
        Node* newNode = new Node(val);
        if (head == nullptr) {
            head = newNode;
            return;
        }
        Node* current = head;
        while (current->next != nullptr) {
            current = current->next;
        }
        current->next = newNode;
    }

    // Function to print the list
    void display() {
        Node* current = head;
        while (current != nullptr) {
            std::cout << current->data << " -> ";
            current = current->next;
        }
        std::cout << "nullptr" << std::endl;
    }

    // Function to delete a node with a specific value
    void deleteNode(int val) {
        Node* current = head;
        Node* prev = nullptr;

        // Case 1: The head node is the one to be deleted
        if (current != nullptr && current->data == val) {
            head = current->next;
            delete current;
            return;
        }

        // Case 2: Search for the node to be deleted
        while (current != nullptr && current->data != val) {
            prev = current;
            current = current->next;
        }

        // If the value was not found
        if (current == nullptr) {
            std::cout << "Value not found." << std::endl;
            return;
        }

        // Unlink the node from the list and deallocate memory
        prev->next = current->next;
        delete current;
    }
    
    // Destructor to free memory
    ~LinkedList() {
        Node* current = head;
        while (current != nullptr) {
            Node* next = current->next;
            delete current;
            current = next;
        }
        head = nullptr;
    }
};

int main() {
    LinkedList list;
    
    // Append elements to the list
    list.append(10);
    list.append(20);
    list.append(30);
    
    std::cout << "Initial list: ";
    list.display(); // Expected: 10 -> 20 -> 30 -> nullptr
    
    // Delete a node
    list.deleteNode(20);
    
    std::cout << "List after deleting 20: ";
    list.display(); // Expected: 10 -> 30 -> nullptr
    
    // Try to delete a value that doesn't exist
    list.deleteNode(99);
    
    return 0;
}