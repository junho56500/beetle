#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>


using namespace std;

int main() {
    std::map<int, std::string> students;
    
    // Insert elements. They are automatically sorted by key.
    students[3] = "Charlie";
    students[1] = "Alice";
    students[2] = "Bob";
    
    // Access and iterate in sorted order
    std::cout << "Sorted by key (std::map):" << std::endl;
    for (const auto& pair : students) {
        std::cout << "ID " << pair.first << ": " << pair.second << std::endl;
    }
    
    // Find an element (logarithmic time)
    auto it = students.find(2);
    if (it != students.end()) {
        std::cout << "Found student with ID 2: " << it->second << std::endl;
    }

    std::set<std::string> uniqueNames;
    
    // Insert unique names. Duplicates are ignored.
    uniqueNames.insert("David");
    uniqueNames.insert("Charlie");
    uniqueNames.insert("David"); // Ignored
    uniqueNames.insert("Alice");
    
    // Iterate in sorted order
    std::cout << "Sorted unique names (std::set):" << std::endl;
    for (const auto& name : uniqueNames) {
        std::cout << name << std::endl;
    }

    auto it2 = uniqueNames.find("David");
    cout << it2->data();

    std::unordered_map<int, std::string> students2;
    
    // Insert elements. Order is not guaranteed.
    students2[3] = "Charlie";
    students2[1] = "Alice";
    students2[2] = "Bob";
    
    // The order is arbitrary
    std::cout << "Unordered by key (std::unordered_map):" << std::endl;
    for (const auto& pair : students2) {
        std::cout << "ID " << pair.first << ": " << pair.second << std::endl;
    }
    
    // Find an element (average constant time)
    auto it4 = students2.find(2);
    if (it4 != students2.end()) {
        std::cout << "Found student with ID 2: " << it->second << std::endl;
    }

    std::unordered_set<std::string> uniqueNames2;
    
    // Insert unique names. Order is not guaranteed.
    uniqueNames2.insert("David");
    uniqueNames2.insert("Charlie");
    uniqueNames2.insert("David"); // Ignored
    uniqueNames2.insert("Alice");
    uniqueNames2.erase("Alice");
    uniqueNames2.insert("Alice");
    
    // The order is arbitrary
    std::cout << "Unordered unique names (std::unordered_set):" << std::endl;
    for (const auto& name : uniqueNames2) {
        std::cout << name << std::endl;
    }

}