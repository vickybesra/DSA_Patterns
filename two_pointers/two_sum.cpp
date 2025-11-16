#include<bits/stdc++.h>
using namespace std;
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
         
        for (int i = 0; i < nums.size(); ++i) {
            int need = target - nums[i];
            if (pos.count(need)) return {pos[need], i};
            pos[nums[i]] = i;
        }
        return {-1,-1};
    }
};
