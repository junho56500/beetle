#include <vector>
#include <cstdint>
#include <cmath>

using namespace std;

constexpr int NUM_OBJECT = 20;
constexpr int NUM_PTS = 20;

struct point{
    float x,y,z;
};
struct BB{
    float xmin, xmax, ymin;
};

struct BB{
    float x,y,z,width;
};

struct Obj{
    uint64_t timestamp;
    uint32_t id;
    
    //representation
    vector<point> pts;  //relative position
    vector<point> BB;
    
    float roll;
    float pitch;
    float yaw;
    
    //movement state
    float speed;
    float angle;

    uint32_t object_type;

    float confidence;
    float probabilty;

    Obj() : pts(NUM_OBJECT,{0,0,0}) {}

};

int main()
{
    
}