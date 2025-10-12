class Solution {
private:
    vector<int> _nums;
    
public:
    Solution(vector<int>& nums) {
        _nums = nums;
    }
    
    vector<int> reset() {
        return _nums;
    }
    
    vector<int> shuffle() {
        vector<int>temp = _nums;
        random_device rd;
        default_random_engine rng(rd());
        std::shuffle(temp.begin(), temp.end(), rng);
        return temp;
    }
    
};