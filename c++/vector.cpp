#include "stdoio.h"

using namespace std;

int main() {
    vector<int> a(3,1);

    for (size_t i = 0; i < a.size(); i++){
        a[i]++;
    }

    for (auto& num:a){
        num++;
    }

    vector<int> b = {1,2,3,4,5};
    
}