from typing import List
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        if n == 0 or k == 0:
            return 0

        # 如果交易次数超过天数的一半，相当于无限次交易
        if k >= n // 2:
            return sum(max(prices[i+1] - prices[i], 0) for i in range(n - 1))

        dp = [[0, float('-inf')] for _ in range(k + 1)]

        for price in prices:
            for j in range(1, k + 1):
                dp[j][1] = max(dp[j][1], dp[j-1][0] - price)  # 买入
                dp[j][0] = max(dp[j][0], dp[j][1] + price)    # 卖出

        return dp[k][0]
