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
    ListNode* reverseList(const ListNode* head) {
        ListNode* reversed = NULL;
        while(head !=nullptr){
            ListNode* temp = new ListNode(head->val);
            if(reversed == NULL)
            {
                reversed = temp;
            }
            else
            {
                temp->next = reversed;
                reversed = temp;
            }
            head = head->next;
        }
        return reversed;
    }
};