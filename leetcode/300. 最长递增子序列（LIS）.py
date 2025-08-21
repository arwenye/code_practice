from typing import List

class Solution:
    def lengthOfLIS1(self, nums: List[int]) -> int:
        '''动态规划解法，O(n^2)'''
        if not nums:
            return 0
        
        n=len(nums)
        dp=[1]*n
        for i in range(1,n):
            for j in range(i):
                if nums[j]<nums[i]:
                    dp[i]=max(dp[j]+1,dp[i])

        return max(dp)
    

    def lengthOfLIS2(self, nums: List[int]) -> int:
        '''贪心+二分，O(nlongn)'''
        from bisect import bisect_left
        if not nums:
            return 0
        sub=[]
        for num in nums:
            idx=bisect_left(sub,num)
            if idx==len(sub):
                sub.append(num)
            else:
                sub[idx]=num
        return len(sub)