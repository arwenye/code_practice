from typing import List

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []  # 结果存储所有子集
        path = []  # 当前构造的子集
        
        def backtrack(start: int):
            res.append(path[:])  # 每次递归都记录当前子集
            for i in range(start, len(nums)):  # 从 start 位置开始
                path.append(nums[i])  # 选择当前元素
                backtrack(i + 1)  # 递归选择下一个
                path.pop()  # 撤销选择
        
        backtrack(0)  # 从第 0 个元素开始
        return res

    def subsets2(self,nums: List[int]) -> List[List[int]]:
        '''迭代法'''
        res = [[]]  # 初始化空子集
        for num in nums:
            res += [subset + [num] for subset in res]  # 每个已有子集都加上 num 形成新子集
        return res

    def subsets3_1(self,nums: List[int]) -> List[List[int]]:
        '''位运算，简单版'''
        n=len(nums)
        res=[]
        for mask in range(1<<n):
            sub=[nums[i] for i in range(n) if (mask&(1<<i))]
            res.append(sub)
        return res
    def subsets3_2(self, nums: List[int]) -> List[List[int]]:
        '''位运算，详细版'''
        n = len(nums)
        res = []
        # 遍历所有可能的子集（从 0 到 2^n-1）
        for mask in range(1 << n):  # 1 << n 代表 2^n
            subset = []
            for i in range(n):  # 遍历 nums 的每个元素
                if mask & (1 << i):  # 检查第 i 位是否为 1
                    subset.append(nums[i])  # 选 nums[i]
            res.append(subset)
        return res
