#include <iostream>
#include <vector>
#include <memory>


using namespace std;

struct PcDataPose {
    double x;
    double y;
    double z;
};

class IPc{
public:    
    virtual ~IPc() = default;
    virtual void print() const = 0;
};

class Pc:public IPc{
public:
    Pc(PcDataPose pc):_pc(pc) {};
    void print() const override {
        cout << _pc.x << _pc.y << _pc.z;
    }

protected:
    PcDataPose _pc;    
};


class Inten:public Pc{
public:
    Inten(PcDataPose pc, int inten):Pc(pc), _inten(inten) {};
    void print() const override {
        Pc::print();
        cout << _inten << endl;
    }
protected:
    int _inten;
};

class Man{
public:
    int addData(double x, double y, double z) {
        _arrPc.push_back(make_shared<Pc>(PcDataPose{x,y,z}));
        return _arrPc.size()-1;
    }
    int addData(double x, double y, double z, int inten){
        _arrPc.push_back(make_shared<Inten>(PcDataPose{x,y,z}, inten));
        return _arrPc.size()-1;
    }
    void printAll()
    {
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
    Man dm;
    dm.addData(1,2,3);
    dm.printAll();
    dm.addData(4,5,6,7);
    dm.printAll();

    return 0;
}