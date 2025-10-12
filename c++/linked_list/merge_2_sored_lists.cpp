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
    
    void printNode(ListNode* a)
    {
        while (a != nullptr)
        {
            cout << a->val << "->";
            a = a->next;
        }
        cout << endl;
    }
    
    ListNode* mergeTwoLists(const ListNode* list1, const ListNode* list2) {
        ListNode* merged = NULL;
        
        if ((list1 == nullptr)&&(list2==nullptr))
        {
            return merged;
        }
        else if(list2 == nullptr)
        {
            merged = new ListNode(list1->val);
            list1 = list1->next;
        }
        else if(list1 == nullptr)
        {
            merged = new ListNode(list2->val);
            list2 = list2->next;
        }
        else
        {
            if(list1->val < list2->val)
            {
                merged = new ListNode(list1->val);
                list1 = list1->next;
            }
            else
            {
                merged = new ListNode(list2->val);
                list2 = list2->next;
            }
        }
        
        
        ListNode* current = merged;
        
        while ((list1 != nullptr) || (list2 != nullptr))
        {
            if(list2 == nullptr)
            {
                current->next = new ListNode(list1->val);
                current = current->next;
                list1 = list1->next;
            }
            else if(list1 == nullptr)
            {
                current->next = new ListNode(list2->val);
                current = current->next;
                list2 = list2->next;
            }
            else
            {
                if(list1->val < list2->val)
                {
                    ListNode* temp = new ListNode(list1->val);
                    current->next = temp;
                    current = current->next;
                    list1 = list1->next;
                    printNode(merged);
                }
                else
                {
                    ListNode* temp = new ListNode(list2->val);
                    current->next = temp;
                    current = current->next;

                    list2 = list2->next;
                    printNode(merged);
                }
            }
            
        }
        return merged;
        
    }
};