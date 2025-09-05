from typing import List
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        res = [0] * n  # 初始化结果数组，默认值为 0
        stack = []  # 使用栈来存储索引
        
        for i in range(n):
            # 当栈非空并且当前温度大于栈顶索引的温度时
            while stack and temperatures[i] > temperatures[stack[-1]]:
                # 弹出栈顶元素，获取该元素的索引
                idx = stack.pop()
                # 计算当前索引和弹出元素的索引之差，即天数
                res[idx] = i - idx
            # 将当前索引压入栈中
            stack.append(i)
        
        return res
