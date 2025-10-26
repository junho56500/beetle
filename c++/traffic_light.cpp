#include <iostream>
#include <memory>
#include <time>

using namespace std;
enum stats{
    NONE,
    GREEN,
    YELLOW,
    RED,
    ARROW,
    NUMOFSTATE
};

struct loc{
    float x;
    float y;
    float z;
};

class TL
{
public:
    TL(loc coords, int id):_coords(coords), _id(id){}
    virtual ~TL() = default; 

    void set(stats state);
    stats get(){
        return _light;
    }
    void display(){
        cout << "current status : "_light;
    }

private:
    stats _light;
    int _arrow[3];
    loc _coords;
    float _height;
    int _id;
};


int main()
{
    //initialization
    shared_ptr<TL> tl1 = make_shared<TL>({128.0, 26.0,35.0}, 1);

    //state machine
    //make threds
    while(1)
    {
        sleep(100);
        tl1.set(GREEN);
    }

    cout << tl1.get();
}