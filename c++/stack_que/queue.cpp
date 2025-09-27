#include <iostream>

using namespace std;

class Queue{
private:
    static const int MAX_SIZE = 100;
    int arr[MAX_SIZE];
    int front;
    int end;

public:
    Queue() : front(-1), end(-1) {};        
    virtual ~Queue() = default;

    bool isEmpty()
    {
        return (front == -1);
    }

    bool isFull()
    {
        return (end == MAX_SIZE - 1);
    }

    void enqueue(int val)
    {
        //check full

        if(isEmpty())
        {
            front = end = 0;
        }
        else
        {
            end++;
        }

        arr[end] = val; 
    }

    void dequeue()
    {
        //check empty
        if(front == end)
        {
            front = end = -1;
        }
        else{
            front++;
        }
    }

    int front()
    {
        //check empty

        return (arr[front]);
    }

};