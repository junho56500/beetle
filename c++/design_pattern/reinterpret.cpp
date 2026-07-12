#include <vector>
#include <iostream>
#include <memory>

using namespace std;

struct c1{
    uint32_t a;
    uint32_t b;
};

struct c2{
    int32_t a;
    int32_t b;
};

void func(void* a)
{
    c1* b = reinterpret_cast<c1*>(a);
    cout << b->a << endl;
    cout << b->b << endl;

    c2* c = reinterpret_cast<c2*>(b);
    cout << c->a << endl;
    cout << c->b << endl;

}

int main()
{
    int64_t buf = 3;
    func(&buf);

    return 0;
}