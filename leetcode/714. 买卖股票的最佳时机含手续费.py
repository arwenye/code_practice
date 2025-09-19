'''含有手续费'''
from typing import List

class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        if not prices:
            return 0
        
        n = len(prices)
        not_hold, hold = 0, -prices[0]
        
        for price in prices[1:]:
            not_hold = max(not_hold, hold + price - fee)  # 卖出
            hold = max(hold, not_hold - price)            # 买入
        
        return not_hold