#include <iostream>
#include <vector>
#include <memory>

class Ipc {
public:
    virtual ~Ipc {} = default
    virtual print() = 0;
}