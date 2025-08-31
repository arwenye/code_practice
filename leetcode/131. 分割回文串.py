from typing import List

class Solution:
    def partition(self, s: str) -> List[List[str]]:
        '''经典回溯'''
        res = []  # 结果集
        path = []  # 当前路径

        # 判断是否是回文
        def isPalindrome(sub: str) -> bool:
            return sub == sub[::-1]

        # 回溯
        def backtrack(start: int):
            if start == len(s):  # 如果遍历到字符串末尾，记录当前方案
                res.append(path[:])
                return

            for end in range(start, len(s)):  # 枚举所有可能的切割点
                substring = s[start:end + 1]
                if isPalindrome(substring):  # 只在是回文时继续搜索
                    path.append(substring)  # 选择
                    backtrack(end + 1)  # 递归探索剩余部分
                    path.pop()  # 撤销选择

        backtrack(0)
        return res
    from typing import List

class Solution:
    def partition(self, s: str) -> List[List[str]]:
        '''使用dp优化计算时间，空间换时间'''
        n = len(s)
        res = []
        path = []

        # 1. 预处理 DP 表
        dp = [[False] * n for _ in range(n)]
        for right in range(n):
            for left in range(right + 1):
                if s[left] == s[right] and (right - left <= 2 or dp[left + 1][right - 1]):
                    dp[left][right] = True  # 标记回文

        # 2. 回溯
        def backtrack(start: int):
            if start == n:  # 终止条件
                res.append(path[:])
                return

            for end in range(start, n):
                if dp[start][end]:  # 只在是回文时递归
                    path.append(s[start:end + 1])
                    backtrack(end + 1)
                    path.pop()

        backtrack(0)
        return res