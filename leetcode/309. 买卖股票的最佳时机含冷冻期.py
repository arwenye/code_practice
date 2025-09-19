from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0

        s0, s1, s2 = 0, -prices[0], 0

        for price in prices[1:]:
            prev_s0, prev_s1, prev_s2 = s0, s1, s2
            s0 = max(prev_s0, prev_s2)          # 不持股，可买入
            s1 = max(prev_s1, prev_s0 - price)  # 买入或继续持股
            s2 = prev_s1 + price                # 卖出进入冷冻期

        return max(s0, s2)  # 最终利润必然不持股
