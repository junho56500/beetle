#include <vector>
#include <span>
#include <unordered_map>
#include <cstddef>

using namespace std;

vector<int> find_common_elements(span<int const> input){
    auto map = unordered_map<int, size_t>();
    for (auto const val: input)
    {
        ++map[val];
    }
    auto res = vector<int>();
    for (auto const val: map){
        if (val.second > input.size()/3){
            res.push_back(val.first);
        }
    }
    return res;

}