
from typing import List
class Solution:
    def trap(height: List[int]) -> int:
        '''暴力解法'''
        n = len(height)
        if n < 3:
            return 0  # 少于3根柱子无法接水
        total = 0
        for i in range(1, n-1):
            # 计算左侧最高
            left_max = max(height[:i+1])
            # 计算右侧最高
            right_max = max(height[i:])
            # 累加当前柱子的接水量
            total += max(0, min(left_max, right_max) - height[i])
        return total
    def trap(self, height: List[int]) -> int:
        '''dp优化lm,rm的计算'''
        n = len(height)
        if n < 3:
            return 0
            
        lm = [0] * n
        rm = [0] * n
        res = 0
        
        # 优化左侧最大值计算
        lm[0] = height[0]
        for i in range(1, n):
            lm[i] = max(lm[i-1], height[i])
        
        # 优化右侧最大值计算
        rm[-1] = height[-1]
        for i in range(n-2, -1, -1):
            rm[i] = max(rm[i+1], height[i])
        
        # 计算总雨水
        for i in range(n):
            res += max(min(lm[i], rm[i]) - height[i], 0)
        
        return res
    
    def trap(height: List[int]) -> int:
        '''双指针，优化时间复杂度'''
        n = len(height)
        if n < 3:
            return 0
        l, r = 0, n-1
        left_max, right_max = height[0], height[-1]
        total = 0
        while l < r:
            if left_max < right_max:#height[l] < height[r] 这个条件的本质是：当前右指针位置的柱子，比左指针位置的柱子高。
                # 左指针移动
                l += 1
                if height[l] > left_max:
                    left_max = height[l]
                else:
                    total += left_max - height[l]
            else:
                # 右指针移动
                r -= 1
                if height[r] > right_max:
                    right_max = height[r]
                else:
                    total += right_max - height[r]
        return total
    
        
