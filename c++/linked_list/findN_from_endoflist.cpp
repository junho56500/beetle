#include <iostream>

// Definition for a singly-linked list node.
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    ListNode* findNthFromEnd(ListNode* head, int n) {
        if (head == nullptr || n <= 0) {
            return nullptr; // Handle edge cases
        }
        
        ListNode* fast_ptr = head;
        ListNode* slow_ptr = head;
        
        // Advance the fast_ptr by n nodes
        for (int i = 0; i < n; ++i) {
            if (fast_ptr == nullptr) {
                return nullptr; // List is shorter than n nodes
            }
            fast_ptr = fast_ptr->next;
        }
        
        // Move both pointers until fast_ptr reaches the end
        while (fast_ptr != nullptr) {
            slow_ptr = slow_ptr->next;
            fast_ptr = fast_ptr->next;
        }
        
        return slow_ptr;
    }
};

// Main function to demonstrate the code
int main() {
    // Create a sample linked list: 1 -> 2 -> 3 -> 4 -> 5
    ListNode* head = new ListNode(1);
    head->next = new ListNode(2);
    head->next->next = new ListNode(3);
    head->next->next->next = new ListNode(4);
    head->next->next->next->next = new ListNode(5);
    
    Solution solution;
    int n = 2; // Find the 2nd node from the end
    
    ListNode* result = solution.findNthFromEnd(head, n);
    
    if (result != nullptr) {
        std::cout << "The " << n << "th node from the end is: " << result->val << std::endl;
    } else {
        std::cout << "The list is too short or n is invalid." << std::endl;
    }
    
    return 0;
}