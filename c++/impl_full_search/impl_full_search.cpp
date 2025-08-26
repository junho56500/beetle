// You are given a list of positive integers nums and a positive integer target. Your task is to find the maximum possible sum of a combination of numbers from nums that is less than or equal to target. Each number in nums can be used at most once in any combination.

// The combination can be of any size, from one number to all the numbers in nums.

// Write a function find_max_combination_sum(nums, target) that returns this maximum sum.

// Examples:

// 1. Input: nums = [1, 2, 3], target = 4
//    Output: 4
//    Explanation: The possible combinations are:
//    [1] -> sum = 1
//    [2] -> sum = 2
//    [3] -> sum = 3
//    [1, 2] -> sum = 3
//    [1, 3] -> sum = 4
//    [2, 3] -> sum = 5 (too large)
//    [1, 2, 3] -> sum = 6 (too large)
//    The sums that are <= 4 are 1, 2, 3, 3, 4. The maximum of these is 4.

// 2. Input: nums = [10, 20, 30], target = 25
//    Output: 20
//    Explanation:
//    [10] -> sum = 10
//    [20] -> sum = 20
//    [30] -> sum = 30 (too large)
//    [10, 20] -> sum = 30 (too large)
//    The maximum sum <= 25 is 20.

// 3. Input: nums = [5, 6, 7], target = 3
//    Output: 0
//    Explanation: No single number or combination is less than or equal to 3. The sum should be initialized to 0.

// 4. Input: nums = [1, 1, 1], target = 2
//    Output: 2
//    Explanation:
//    [1] -> sum = 1
//    [1, 1] -> sum = 2
//    [1, 1, 1] -> sum = 3 (too large)
//    The maximum sum <= 2 is 2.

#include "stdio.h"
#include <iostream>
#include <vector>

using namespace std;

int findCombination(const vector<int>& nums, int target)
{
    int n = nums.size();
    int n_subset = 1 << n;
    int max_sum = 0;

    for(int i = 0; i< n_subset; i++)
    {
        int sum = 0;
        for(int j=0; j < n; j++)
        {
            if ((i >> j) & 1)
            {
                sum += nums[j];
            }
        }
        if(sum <= target)
        {
            max_sum = max(max_sum, sum);
        }
    }
    return max_sum;
}

int main() {
    vector<int> nums = {};
    int target = 0;

    int input = 0;
    while (cin >> input && input > 0){
        nums.push_back(input); 
    }

    cout << "you input " << nums.size() << "\n";

    cin >> target;

    int ret = 0;
    if(target > 0 && sizeof(nums) > 0){
        ret = findCombination(nums, target);
    }

    cout << ret;
    return 0;

}