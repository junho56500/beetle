#include <iostream>
#include <vector>
#include <memory>


using namespace std;

struct PcDataPose {
    double x;
    double y;
    double z;
};

struct PcDataAngle {
    double yaw;
    double pitch;
    double roll;
};


class IPc{
public:    
    virtual void print() const = 0;
    virtual ~IPc() = default;
};

class Pc : public IPc{
public:
    Pc(PcDataPose pcData):_pcData(pcData) {};
    void print() const override {
        cout << _pcData.x << _pcData.y << _pcData.z;
    }

protected:
    PcDataPose _pcData;
};

class Inten : public Pc{
public:
    Inten (PcDataPose pcData, int inten):Pc(pcData), _inten(inten) {};
    void print() const override {
        Pc::print();
        cout << _inten << endl;
    }
protected:
    int _inten;
};

class Angle : public Pc{
public:    
    Angle (PcDataPose pcData, PcDataAngle pcData2):Pc(pcData), _pcData2(pcData2) {};
    void print() const override
    {
        Pc::print();
        cout << _pcData2.yaw << _pcData2.pitch << _pcData2.roll << endl;
    } 

protected:
    PcDataAngle  _pcData2;
};

class Man{
public:
    int addData(const double& x, const double& y, const double& z ){
        _arrPc.push_back(make_shared<Pc>(PcDataPose{x,y,z}));
        return _arrPc.size() - 1;
    }
    int addData(const double& x, const double& y, const double& z, const int& inten)
    {
        _arrPc.push_back(make_shared<Inten>(PcDataPose{x,y,z},inten));
        return _arrPc.size() - 1;
    }
    int addData(const double& x, const double& y, const double& z, const double& yaw, const double& pitch, const double& roll)
    {
        _arrPc.push_back(make_shared<Angle>(PcDataPose{x, y, z}, PcDataAngle{yaw, pitch, roll}));
        return _arrPc.size() - 1;
    }

    int delData (int ind){
        _arrPc.erase(_arrPc.begin() + ind);
        return _arrPc.size() - 1;
    }

    void printAll() const
    {
        for(const auto &i: _arrPc)
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
    dm.addData(1,2,3,4);
    dm.addData(1,2,3,4,5,6);
    dm.printAll();
    dm.delData(2);
    dm.printAll();
    return 0;
}
