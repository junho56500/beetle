#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <string>
#include <stdexcept>

using namespace std;

class IPc{
public:
    virtual ~IPc () = default; 
    virtual void print() const = 0;

};

class Pc : public IPc{
public:
    Pc(double x, double y, double z):_x(x), _y(y), _z(z) {}
    void print() const override
    {
        cout << _x << _y << _z;
    } 

protected:
    double _x, _y, _z;

};

class Inten : public Pc{
public:
    Inten(double x, double y, double z, int inten):Pc(x, y, z), _inten(inten) {}
    void print() const override{
        Pc::print();
        cout << _inten; 
    }

protected:
    int _inten;
};

class Man{
public:
    int addPc(double x, double y, double z)
    {
        _arrPc.push_back(make_shared<Pc>(x, y, z));
        return _arrPc.size() - 1;
    }

    int addInten(double x, double y, double z, int inten)
    {
        _arrPc.push_back(make_shared<Inten>(x,y,z,inten));
        return _arrPc.size() -1;
    }

    int delPc(int idx)
    {
        _arrPc.erase(_arrPc.begin()+idx);
        return _arrPc.size() - 1;
    }

    int printAll() const
    {
        for (const auto& i: _arrPc)
        {
            i->print();
            cout << '\n';
        }
    }

private:
    vector<shared_ptr<Pc>> _arrPc;
};

int main()
{
    Man pcManager;
    pcManager.addPc(2.1, 2.3, 4.2);
    pcManager.addInten(2.3, 4.2, 5, 3);
    pcManager.printAll();

    pcManager.delPc(0);
    pcManager.printAll();

    return 0;
}

