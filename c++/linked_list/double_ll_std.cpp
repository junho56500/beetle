#include <iostream>
#include <list>
#include <string>
#include <numeric> // For std::accumulate

// Feature	            std::list	                            std::vector
// Data Structure	    Doubly Linked List	                    Dynamic Array
// Memory	            Non-contiguous (nodes are spread out)	Contiguous (single block)
// Insertion/Deletion	Fast (O(1)) at any position	            Slow (O(n)) in the middle, fast (O(1) amortized) at the end
// Random Access	    Slow (O(n)), requires traversal	        Fast (O(1)), using [] operator
// Memory Overhead	    High (each element has two pointers)	Low (just the data)

int main() {
    // 1. Create a std::list of integers
    std::list<int> myList;

    // 2. Add elements to the list
    myList.push_back(10);  // Add to the end
    myList.push_front(5);  // Add to the beginning
    myList.push_back(20);

    // 3. Insert an element in the middle
    // You need an iterator to point to the position
    auto it = myList.begin();
    std::advance(it, 2); // Move iterator forward by 2 positions

    // Access the value at the third position
    std::cout << "The third item is: " << *it << std::endl;

    myList.insert(it, 15);

    // 4. Print the list
    std::cout << "List contents: ";
    for (const auto& val : myList) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 5. Remove an element
    myList.remove(15); // Removes all occurrences of 15
    myList.erase(it);

    // 6. Print the list after removal
    std::cout << "List after removing 15: ";
    for (const auto& val : myList) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 7. Check the size of the list
    std::cout << "List size: " << myList.size() << std::endl;

    // 8. Access first and last elements
    std::cout << "First element: " << myList.front() << std::endl;
    std::cout << "Last element: " << myList.back() << std::endl;

    // 9. Sort the list
    myList.sort();
    std::cout << "Sorted list: ";
    for (const auto& val : myList) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}