
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        '''贪心算法'''
        ans=nums[0]
        sum=0
        for i in range(len(nums)):
            if sum>0:
                sum+=nums[i]
            else:
                sum=nums[i]
            if sum>ans:
                ans=sum
        return ans
    
    def maxSubArray2(nums):
        '''前缀和，当前减去历史最小'''
        s = 0
        m = 0
        ans = float('-inf')
        for x in nums:
            s += x
            ans = max(ans, s - m)
            m = min(m, s)
        return ans
    
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0

        dp = [0] * n
        dp[0] = nums[0]  # 初始化第一个位置
        
        for i in range(1, n):
            # 状态转移：要么接着前面的子数组，要么从当前开始
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
            

        return max(dp)
