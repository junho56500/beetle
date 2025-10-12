/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    
    void printList(ListNode* head) {
        ListNode* current = head;
        while (current != NULL) {
            std::cout << current->val << " -> ";
            current = current->next;
        }
        std::cout << "NULL" << std::endl;
    }
    
    void deleteNode(ListNode* node) {
        node->val = node -> next -> val;
        node->next = node->next->next;
        
        printList(node);
    }
};

//Remove Nth node from end of list

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        
        if(head->next == nullptr)
        {
            cout << "nullptr " << head->val;
            delete head;
            return NULL;
        }
            
        ListNode* fast = head;
        ListNode* slow = head;
        
        for (int i = 0; i < n; i++)
        {
            fast = fast->next;
        }
        
        ListNode* prev;
        
        while(fast != nullptr)
        {
            prev = slow;
            slow = slow->next;
            fast = fast->next;
        }
        
        if (slow->next == nullptr)
        {
            prev->next = nullptr;
            delete slow;
        }
        else
        {
            slow->val = slow->next->val;
            slow->next = slow->next->next;
        }
        return head;
    }
};