
from typing import List

class Solution:
    def maximumProfit(self, prices: List[int], k: int) -> int:
        if not prices:
            return 0

        dp = [[0, -float('inf'), -float('inf')] for _ in range(k+1)]
        for price in prices:
            prev = [row[:] for row in dp]
            for j in range(1, k+1):
                dp[j][0] = max(prev[j][0], prev[j][1] + price, prev[j][2] - price)
                dp[j][1] = max(prev[j][1], prev[j-1][0] - price)
                dp[j][2] = max(prev[j][2], prev[j-1][0] + price)

        return max(dp[j][0] for j in range(k+1))
