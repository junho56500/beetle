#include "stdio.h"
#include <vector>
#include <string>
#include <iostream> // For output
#include <algorithm> // For std::sort, std::find, etc.

using namespace std;

int main() {
    vector<int> a(3,1);

    for (size_t i = 0; i < a.size(); i++){
        a[i]++;
    }

    for (auto& num:a){
        num++;
    }

    std::vector<int> myVector; // An empty vector of integers
    std::vector<double> scores(10); // A vector of 10 doubles, initialized to 0.0
    std::vector<std::string> names(5, "Unknown"); // A vector of 5 strings, all "Unknown"
    std::vector<char> letters = {'a', 'b', 'c'}; // Initializer list (C++11+)
    std::vector<int> anotherVector(myVector); // Copy constructor
    std::vector<int> subVector(myVector.begin() + 1, myVector.end() - 1); // From iterators
    
    for (int x : myVector) {
        std::cout << x << " ";
    }

    if (myVector.empty()) {
        std::cout << "Vector is empty." << std::endl;
    }

    myVector.push_back(100);
    myVector.emplace_back(200);
    myVector.pop_back();

    myVector.insert(myVector.begin() + 1, 50); // Insert 50 at index 1
    myVector.insert(myVector.end(), {300, 400, 500}); // Insert multiple

    myVector.erase(myVector.begin() + 1); // Erase element at index 1
    myVector.erase(myVector.begin(), myVector.begin() + 2); // Erase first two elements

    std::vector<int> v1 = {1, 2};
    std::vector<int> v2 = {3, 4, 5};
    v1.swap(v2); // v1 is now {3,4,5}, v2 is {1,2}
}