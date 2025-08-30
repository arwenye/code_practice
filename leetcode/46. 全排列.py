from typing import List

class Solution:
    def permute1(self, nums: List[int]) -> List[List[int]]:
        '''取数构造每一个位置，需要标识数组'''
        res = []
        used = [False] * len(nums)  # 记录哪些数已经被用过
        path = []  # 当前排列路径

        def backtrack():
            # 递归终止条件
            if len(path) == len(nums):
                res.append(path[:])  # 复制当前排列并加入结果集
                return
            
            for i in range(len(nums)):  # 遍历所有元素
                if not used[i]:  # 如果该元素未被使用
                    path.append(nums[i])  # 选择当前元素
                    used[i] = True  # 标记已使用
                    backtrack()  # 递归进入下一层
                    path.pop()  # 撤销选择（回溯）
                    used[i] = False  # 取消标记

        backtrack()
        return res
    
    def permute2(self, nums: List[int]) -> List[List[int]]:
        '''通过换位置，依次处理，节省空间'''
        res = []

        def backtrack(start):
            if start == len(nums):  # 终止条件
                res.append(nums[:])
                return
            
            for i in range(start, len(nums)):  # 交换 `start` 和 `i`
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1)  # 递归处理后续元素
                nums[start], nums[i] = nums[i], nums[start]  # 回溯

        backtrack(0)
        return res
            
    