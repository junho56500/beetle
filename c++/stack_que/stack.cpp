#include <iostream>

using namespace std;

class Stack{
private:
    static const int MAX_SIZE = 100;
    int arr[MAX_SIZE];
    int top;

public:
    Stack() : top(-1) {}

    virtual ~Stack() = default;

    bool isEmpty() {
        return (top == -1);
    }

    bool isFull() {
        return (top == MAX_SIZE - 1);
    }

    void push(int val)
    {
        if(isFull())
        {
            cout << "stack is full" << endl;
            return;
        }
        
        arr[++top] = val;
    }

    void pop()
    {
        if(isEmpty())
        {
            cout << "already empty" << endl;
            return;
        }
        top--;
    }

    int top()
    {
        if(isEmpty())
        {
            cout << "already empty" << endl;
            return;
        }
        return arr[top];
    }
};