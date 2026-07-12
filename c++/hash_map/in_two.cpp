#include <vector>
#include <span>
#include <unordered_map>
#include <cstddef>

using namespace std;

vector<int> in_tow(span<int const> num1, span<int const> num2, span<int const> num3)
{
    struct found_in {
        bool n1 = false;
        bool n2 = false;
        bool n3 = false;
    };

    auto map = unordered_map<int, found_in>();
    for (auto const val : num1){map[val].n1 = true;}
    for (auto const val : num2){map[val].n2 = true;}
    for (auto const val : num3){map[val].n3 = true;}
    auto res = vector<int>();
    for (auto const x : map){
        if (x.second.n1 + x.second.n2 + x.second.n3 >= 2) {res.push_back(x.first);}
    }
    return res;

}