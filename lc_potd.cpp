class Solution {
public:
    int solve(int i, int j, vector<int>& nums1, vector<int>& nums2, vector<vector<int>>& dp){
        if(i>=(int)nums1.size() or j>=(int)nums2.size()){
            return -1e8; 
        }

        if(dp[i][j]!=-1e9) return dp[i][j] ;

        int ntake1 = solve(i, j+1, nums1, nums2, dp) ;
        int ntake2 = solve(i+1, j, nums1, nums2, dp) ;
        int ntake3 = solve(i+1, j+1, nums1, nums2, dp) ;
        int take = nums1[i]*nums2[j] + max(0,ntake3) ;

        return dp[i][j] = max({take, ntake1, ntake2, ntake3}); 

    }


    int maxDotProduct(vector<int>& nums1, vector<int>& nums2) {
        int n=nums1.size(), m=nums2.size() ;
        vector<vector<int>> dp(n+1, vector<int>(m+1,-1e9));
        return solve(0,0,nums1, nums2, dp) ;
    }
};
