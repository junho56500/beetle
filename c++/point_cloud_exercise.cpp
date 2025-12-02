#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <string>
#include <stdexcept>

using namespace std;

class IPointCloud{
public:
    virtual ~IPointCloud() = default;
    virtual void print() const = 0;
};

class PointData: public IPointCloud{
public:
    PointData(double x, double y, double z):_x(x), _y(y), _z(z){};
    void print() const override {
        cout << _x << _y << _z;
    }
protected:
    double _x, _y, _z;

};

class IntensityData : public PointData{
public:
    IntensityData(double x, double y, double z, int intensity):PointData(x, y, z), _intensity(intensity) {};
    void print() const override {
        PointData::print();
        cout << _intensity;
    }
private:
    int _intensity;
};


class OrientationData : public PointData{
public:
    OrientationData(double x, double y, double z, double roll, double pitch, double yaw):PointData(x, y, z), _roll(roll), _pitch(pitch), _yaw(yaw) {};
    void print() const override {
        PointData::print();
        cout << _roll << _pitch << _yaw;
    }
private:
    double _roll;
    double _pitch;
    double _yaw;
};

class DataManager{
private:
    vector<unique_ptr<PointData>> collection;
public:
    void printAll() {
        for (const auto& data : collection)
        {
            data->print();
            cout << endl;
        }
    }

    size_t addData(const double& x, const double& y, const double& z )
    {
        collection.push_back(make_unique<PointData>(x,y,z));
        return collection.size()-1;
    }

    size_t addData(const double& x, const double& y, const double& z, const int& intensity)
    {
        collection.push_back(make_unique<IntensityData>(x,y,z,intensity));
        return collection.size()-1;
    }

    size_t addData(const double& x, const double& y, const double& z, const double& roll, const double& pitch, const double& yaw)
    {
        collection.push_back(make_unique<OrientationData>(x,y,z,roll,pitch,yaw));
        return collection.size()-1;
    }

    bool deleteData(int idx)
    {
        collection.erase(collection.begin()+idx);
        return true;
    }

};

int main()
{
    DataManager dm;
    dm.addData(1,2,3);
    dm.addData(1,2,3,4);
    dm.addData(1,2,3,4,5,6);
    dm.printAll();
    dm.deleteData(2);
    dm.printAll();
    return 0;
}