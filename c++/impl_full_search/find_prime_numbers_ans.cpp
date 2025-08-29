// Problem Description:
// You are given a string of digits, `numbers`, representing pieces of paper.
// You need to find how many prime numbers can be made by combining these
// pieces of paper.
//
// For example, if `numbers` is "17", you can create the numbers 1, 7, 17, and 71.
// Among these, 7, 17, and 71 are prime numbers. So the answer is 3.
//
// Your task is to write a program that finds the total number of unique prime
// numbers that can be formed using the digits in the input string.
//
// Input Conditions:
// - The input will be a string `numbers` consisting of digits (0-9).
// - The length of `numbers` will be between 1 and 7.
//
// Output Conditions:
// - Print the number of unique prime numbers that can be formed.

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;


bool isPrime(const int& num)
{
    cout << num << '\n';
    cout << sqrt(num) << '\n';
    if (num <= 1)
    {
        return false;
    }
    for (int i=2; i<=sqrt(num); i++)
    {
        cout << i << '\n';
        if (num%i == 0)
        {
            return false;
        }
    }
    return true;
}

int find_print_number(const string& nums)
{
    int n = nums.size();

    string temp_nums(nums);

    sort(temp_nums.begin(), temp_nums.end());

    int cnt_prime = 0;
    do{
        for(int i=1; i<temp_nums.size(); i++)
        {
            string sub = temp_nums.substr(0, i);
            int i_sub = stoi(sub);

            if (isPrime(i_sub))
            {
                cnt_prime++;
            }
        }
    }while(next_permutation(temp_nums.begin(), temp_nums.end()));

    return cnt_prime;
}

int main()
{
    string nums;
    cin >> nums;

    int ret = find_print_number(nums);

    cout << ret;

    return 0;
}
