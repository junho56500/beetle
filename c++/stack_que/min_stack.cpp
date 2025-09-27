
// Input
// ["MinStack","push","push","push","getMin","pop","top","getMin"]
// [[],[-2],[0],[-3],[],[],[],[]]

// Output
// [null,null,null,null,-3,null,0,-2]

// Explanation
// MinStack minStack = new MinStack();
// minStack.push(-2);
// minStack.push(0);
// minStack.push(-3);
// minStack.getMin(); // return -3
// minStack.pop();
// minStack.top();    // return 0
// minStack.getMin(); // return -2


class MinStack {
private:
    stack<int> _s;
    stack<int> _min;
    
public:
    MinStack() {
    }
    
    void push(int val) {
        cout << "push() : " << val << endl;
        if(_min.empty() || _min.top() >= val )
        {
            _min.push(val);
        }
        _s.push(val);
    }
    
    void pop() {
        
        cout << "pop() : " << _s.top() << endl;
        if(_min.top() == _s.top())
        {
            _min.pop();
        }
        _s.pop();
    }
    
    int top() {
        
        cout << "top() : " << _s.top() << endl;
        return _s.top();
    }
    
    int getMin() {
        
        cout << "getMin() : " << _min.top() << endl;
        return _min.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(val);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */