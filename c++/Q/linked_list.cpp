//remove previous list if the current and next value is same
#include <iostream>

struct Node {
    int data;
    Node* next;
};

// Function to remove the previous node if current and next have the same value
void remove_previous_if_same_next(Node** head_ref) {
    if (*head_ref == nullptr || (*head_ref)->next == nullptr || (*head_ref)->next->next == nullptr) {
        return; // Not enough nodes to perform the check
    }

    Node* current = *head_ref;
    Node* prev = nullptr;
    Node* prev_of_prev = nullptr;

    while (current->next != nullptr) {
        if (current->data == current->next->data) {
            // Found a case where current and next nodes have the same value
            if (prev_of_prev != nullptr) {
                // If it's not the head node, bypass the 'prev' node
                prev_of_prev->next = current;
                delete prev;
            } else {
                // If the 'prev' node is the head, update the head
                *head_ref = current;
                delete prev;
            }
            // Prev is now removed, so re-align pointers
            prev = prev_of_prev;
        }
        
        // Update pointers for the next iteration
        prev_of_prev = prev;
        prev = current;
        current = current->next;
    }
}



struct Node{
    int val;
    Node* next;
};

void remove(Node** head_ref)
{
    Node* cur(*head_ref);
    Node* prev(nullptr);
    Node* prev_prev(nullptr);

    if(cur->next != nullptr)
    {
        prev = cur;
        cur = cur->next;
    }
    

    while(cur->next != nullptr)
    {
        if (cur->val == cur->next->val)
        {
            if (prev == *head_ref)
            {
                *head_ref = cur;
                delete prev;
            }
            else if(prev_prev != nullptr)
            {
                prev_prev->next = prev->next;
                delete prev;
                prev = prev_prev;
            }
        }

        prev_prev = prev;
        prev = cur;
        cur = cur->next;
    }
}


# 맨처음 list 지울 때는 head = cur 식으로 head 자체를 바꾸어야 함