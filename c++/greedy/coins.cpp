#include <iostream>
#include <vector>
#include <algorithm>


// Given a set of coin denominations (e.g., $1, $5, $10, $25) and a target amount, write code to find the minimum number of coins required to make that amount.

// Function to find the minimum number of coins for a given amount using a greedy approach.
// This greedy algorithm works correctly for standard currency denominations (like USD).
int minCoinsGreedy(int amount, std::vector<int> denominations) {
    // Sort denominations in descending order to apply the greedy strategy effectively.
    // The greedy approach relies on picking the largest available coin first.
    std::sort(denominations.begin(), denominations.end(), std::greater<int>());

    int coin_count = 0;
    
    // Iterate through the sorted denominations.
    for (int coin : denominations) {
        // While the current amount is greater than or equal to the current coin's value,
        // take as many of this coin as possible.
        while (amount >= coin) {
            amount -= coin;
            coin_count++;
        }
    }
    
    // If the remaining amount is 0, we found a valid combination.
    if (amount == 0) {
        return coin_count;
    } else {
        // This case would indicate that the amount cannot be made with the given denominations.
        // For this simple greedy problem, we assume a solution always exists.
        return -1; // Or some other indicator of failure.
    }
}

int main() {
    // Standard US currency denominations.
    // The greedy algorithm works optimally with this set.
    std::vector<int> us_coins = {1, 5, 10, 25};
    
    int amount1 = 63;
    int amount2 = 87;

    int coins_needed1 = minCoinsGreedy(amount1, us_coins);
    int coins_needed2 = minCoinsGreedy(amount2, us_coins);

    std::cout << "For an amount of " << amount1 << ", the minimum coins needed is: "
              << coins_needed1 << std::endl;
    // Expected greedy output: 63 = 25 + 25 + 10 + 1 + 1 + 1 (6 coins)
              
    std::cout << "For an amount of " << amount2 << ", the minimum coins needed is: "
              << coins_needed2 << std::endl;
    // Expected greedy output: 87 = 25 + 25 + 25 + 10 + 1 + 1 (6 coins)
    
    // --- Example where greedy fails (a fun follow-up problem!) ---
    // Consider this set of coins. A greedy approach will not find the optimal solution.
    // The amount of 8 would be 5 + 1 + 1 + 1 (4 coins), but the optimal is 4 + 4 (2 coins).
    std::cout << "\n--- A case where the greedy algorithm fails ---" << std::endl;
    std::vector<int> strange_coins = {1, 4, 5};
    int amount_fail = 8;
    int coins_needed_fail = minCoinsGreedy(amount_fail, strange_coins);
    std::cout << "For an amount of " << amount_fail << ", the greedy algorithm gives "
              << coins_needed_fail << " coins." << std::endl;
    std::cout << "The correct answer is 2 coins (4 + 4)." << std::endl;

    return 0;
}
