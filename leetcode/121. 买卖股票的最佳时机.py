from typing import List
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''贪心算法'''
        if not prices:
            return 0
        min_price = prices[0]  # 初始化最小买入价
        max_profit = 0         # 初始化最大利润

        for price in prices:
            min_price = min(min_price, price)  # 更新最低买入价
            max_profit = max(max_profit, price - min_price)  # 计算当前最大利润

        return max_profit
    
    def maxProfit_dp(self, prices: List[int]) -> int:
        '''动态规划算法，有持有和不持有两种状态'''
        if not prices:
            return 0
        n=len(prices)
        dp=[[0]*2 for _ in range(n)]

        dp[0][0]=0
        dp[0][1]=-prices[0]
        for i in range(1,n):
            dp[i][0]=max(dp[i-1][0],dp[i-1][1]+prices[i])
            dp[i][1]=max(dp[i-1][1],-prices[i])#每一天都尝试重新买入股票，只能拥有一支股票

        return dp[-1][0]
    
    def maxProfit_dp_opt(self, prices: List[int]) -> int:
        '''动态规划算法，滚动数'''
        if not prices:
            return 0
        n=len(prices)
        # 初始化状态：dp_prev_0=前一天不持股，dp_prev_1=前一天持股
        dp_prev_0 = 0
        dp_prev_1 = -prices[0]
        for price in prices[1:]:
            # 计算当天不持股和持股的利润
            dp_curr_0 = max(dp_prev_0, dp_prev_1 + price)
            dp_curr_1 = max(dp_prev_1, -price)  # 仅一次交易，买入时利润为 -price
            # 滚动更新前一天状态
            dp_prev_0, dp_prev_1 = dp_curr_0, dp_curr_1
        return dp_prev_0  # 最终不持股的利润一定大于持股（持股没卖等于没获利）
