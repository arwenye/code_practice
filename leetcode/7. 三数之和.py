
from typing import List

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        nums.sort()
        ans = []

        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:  # 去重
                continue

            l, r = i + 1, n - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]

                if s == 0:
                    ans.append([nums[i], nums[l], nums[r]])

                    while l < r and nums[l] == nums[l+1]:
                        l += 1
                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                    
                    # 移动指针，避免重复计算
                    l += 1
                    r -= 1

                elif s < 0:
                    l += 1
                else:
                    r -= 1
        
        return ans

                
