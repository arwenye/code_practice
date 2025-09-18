from typing import List
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''贪心算法，只要有价钱差就买卖操作'''
        if not prices:
            return 0
        max_profit = 0
        for i in range(1, len(prices)):
            # 累加所有正差价
            if prices[i] > prices[i-1]:
                max_profit += prices[i] - prices[i-1]
        return max_profit
    

    def maxProfit_dp(self, prices: List[int]) -> int:
        '''动态规划，交易通用模板'''
        if not prices:
            return 0
        n=len(prices)
        dp=[[0]*2 for _ in range(n)]
        dp[0][0]=0
        dp[0][1]=-prices[0]

        for i in range(1,n):
            dp[i][0]=max(dp[i-1][0],dp[i-1][1]+prices[i])#卖出
            dp[i][1]=max(dp[i-1][1],dp[i-1][0]-prices[i])#买入

        return dp[-1][0]

    def maxProfit_opt(prices: List[int]) -> int:
        '''空间优化'''
        if not prices:
            return 0
        cash = 0
        hold = -prices[0]

        for price in prices[1:]:
            cash = max(cash, hold + price)
            hold = max(hold, cash - price)

        return cash
    

