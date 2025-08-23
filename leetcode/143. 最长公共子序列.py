class Solution:
    def longestCommonSubsequence1(self, text1: str, text2: str) -> int:
        '''二维数组dp'''
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]  # 创建 DP 表
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:  # 当前字符匹配
                    dp[i][j] = dp[i-1][j-1] + 1
                else:  # 当前字符不匹配
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]  # 返回 LCS 长度

    def longestCommonSubsequence2(self, text1: str, text2: str) -> int:
        '''滚动数组dp'''
        m, n = len(text1), len(text2)
        pre = [0] * (n + 1)
        cur = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    cur[j] = pre[j - 1] + 1  # 继承对角线值 + 1
                else:
                    cur[j] = max(pre[j], cur[j - 1])  # 取上方或左侧较大值
            pre = cur[:]  # ✅ 复制 cur，而不是直接赋值,一定注意啊！！错好几次了
            
        return pre[n]  # 最终答案在 pre 里