#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <string>
#include <stdexcept>

using namespace std;

class Ipc{
public:
    virtual ~Ipc() = default;
    virtual void print() const = 0;
};

class Pc : public Ipc{
public:
    Pc(double x, double y, double z):_x(x),_y(y),_z(z) {}
    void print() const override {
        cout << _x << _y << _z;
    }

protected:
    double _x,_y,_z;
};

class Intensity : public Pc{
public:
    Intensity(double x, double y, double z, int intensity):
        Pc(x,y,z),_intensity(intensity) {}
    void print() const override{
        Pc::print();
        cout << _intensity;
    }

protected:
    int _intensity;
};

class Manager{
public:
    int addPc(double x, double y, double z){
        _arrPc.push_back(make_shared<Pc>(x,y,z));
        return _arrPc.size() - 1;
    }

    int addIntensity(double x, double y, double z, int intensity)
    {
        _arrPc.push_back(make_shared<Intensity>(x, y, z, intensity));
        return _arrPc.size() - 1;
    }

    int delPc(int idx)
    {
        _arrPc.erase(_arrPc.begin()+idx);
        return _arrPc.size() - 1;
    }

    void printAll() const{
        for (const auto& i: _arrPc)
        {
            i->print();
        }
    }

private:
    vector<shared_ptr<Pc>> _arrPc;
};

int main()
{
    Manager a;
    a.addPc(0.1, 0.2, 0.3);
    a.delPc();

}