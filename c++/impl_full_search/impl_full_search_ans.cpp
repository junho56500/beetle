// This program solves the "Maximum Value Combination" problem using an iterative
// brute-force approach with bit manipulation.
// The goal is to find the maximum sum of a subset of numbers from a given list that is
// less than or equal to a specified target value.

#include <iostream>
#include <vector>
#include <numeric> // For std::accumulate
#include <algorithm> // For std::max
#include <cmath> // For std::pow
#include <iterator> // For std::size (C++17+)

// This function finds the maximum combination sum using an iterative approach.
// It uses a bitmask to generate all 2^N subsets of the input vector.
int findMaxCombinationSumIterative(const std::vector<int>& nums, int target) {
    int maxSum = 0;
    int n = nums.size();
    
    // The total number of subsets is 2^n. We use a bitmask from 0 to (2^n) - 1
    // to represent each unique subset. A '1' in the bitmask at a given position
    // means we include the number from the nums vector at that same position.
    long long totalSubsets = 1LL << n;

    // Loop through every possible subset.
    for (long long i = 0; i < totalSubsets; ++i) {
        int currentSum = 0;
        
        // Loop through each element in the input vector.
        for (int j = 0; j < n; ++j) {
            // Check if the j-th bit is set in our current subset number 'i'.
            // The expression (i >> j) checks the j-th bit of i.
            // (i >> j) & 1 will be 1 if the bit is set, and 0 otherwise.
            if ((i >> j) & 1) {
                currentSum += nums[j];
            }
        }
        
        // If the current subset sum is valid (<= target), we update our maxSum.
        if (currentSum <= target) {
            maxSum = std::max(maxSum, currentSum);
        }
    }
    
    return maxSum;
}


// Demonstrates how to pass a C-style array to a function and get its size.
// This is possible because the function template deduces the size N at compile time.
// C++17 introduced std::size to make this even cleaner.
template <typename T, std::size_t N>
void printArrayInfo(const T (&arr)[N]) {
    // The size is known at compile time via the template parameter 'N'.
    std::cout << "Using function template with reference:" << std::endl;
    std::cout << "The size of the array is " << N << std::endl;
    std::cout << "The size (in bytes) of the array is " << sizeof(arr) << std::endl;
    // std::cout << "The size (number of elements) using std::size is " << arr.size() << std::endl;
    std::cout << std::endl;
}

int main() {
    std::vector<int> nums;
    int target;
    int input;

    // Prompt for and read the numbers for the vector.
    std::cout << "Enter a series of positive integers for the 'nums' list." << std::endl;
    std::cout << "Enter -1 to finish entering numbers." << std::endl;
    while (std::cin >> input && input != -1) {
        if (input > 0) { // Only accept positive integers
            nums.push_back(input);
        } else {
            std::cout << "Please enter a positive integer or -1 to stop." << std::endl;
        }
    }

    // Check if any numbers were entered.
    if (nums.empty()) {
        std::cout << "No numbers were entered. Exiting." << std::endl;
        return 0;
    }

    // Clear the input buffer.
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Prompt for and read the target value.
    std::cout << "\nEnter the target value: ";
    std::cin >> target;

    // Call the iterative function to find the maximum combination sum.
    int result = findMaxCombinationSumIterative(nums, target);

    std::cout << "\n-----------------------------------" << std::endl;
    std::cout << "Input nums: [";
    for (size_t i = 0; i < nums.size(); ++i) {
        std::cout << nums[i] << (i == nums.size() - 1 ? "" : ", ");
    }
    std::cout << "], target = " << target << std::endl;
    std::cout << "Output: " << result << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    // Example demonstrating std::size and array reference trick
    int cStyleArray[] = {10, 20, 30, 40, 50};
    printArrayInfo(cStyleArray);

    return 0;
}
