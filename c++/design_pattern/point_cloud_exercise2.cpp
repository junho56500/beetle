#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <string>
#include <stdexcept>

using namespace std;

class IPc{
public:
    virtual ~IPc() = default;
    virtual void print() const = 0;
};

class Pc : public IPc {
public:
    Pc(double x, double y, double z):_x(x), _y(y), _z(z) {}
    void print() const override
    {
        cout << _x << _y << _z;
    }
protected:
    double _x, _y, _z;
};

class IntenPc : public Pc {
public:
    IntenPc(double x, double y, double z, int inten):Pc(x, y, z), _inten(inten) {}
    void print() const override
    {
        Pc::print();
        cout << _inten;
    }
protected:
    int _inten;
};

class manager {
public:
    void printAll() const
    {
        for(const auto& i: _arrPc)
        {
            i->print();
        }
    }

    int addPc(const double& x, const double& y, const double& z) {
        _arrPc.push_back(make_shared<Pc>(x,y,z));
        return _arrPc.size() - 1;
    }

    int addPc(double x, double y, double z, int inten) {
        _arrPc.push_back(make_shared<IntenPc>(x,y,z,inten));
        return _arrPc.size() - 1;
    }

    void delPc(int idx)
    {
        _arrPc.erase(_arrPc.begin() + idx);
    }

private:
    vector<shared_ptr<Pc>> _arrPc;
};


int main()
{
    manager manPc;
    manPc.addPc(0.1, 0.2, 0.3);
    return 0;
}