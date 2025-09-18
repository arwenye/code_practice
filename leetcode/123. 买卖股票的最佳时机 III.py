
from typing import List
class Solution:
    
    def maxProfit(self, prices: List[int]) -> int:
        '''状态dp'''
        if not prices:
            return 0
        
        hold1=float('-inf')
        sell1=0
        hold2=float('-inf')
        sell2=0
        n=len(prices)
        for i in range(n):
            hold1=max(hold1,-prices[i])
            sell1=max(sell1,hold1+prices[i])
            hold2=max(hold2,sell1-prices[i])
            sell2=max(sell2,hold2+prices[i])
        return sell2
    
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n == 0:
            return 0
        left=[0]*n
        right=[0]*n
        min_price=prices[0]
        for i in range(1,n):
            min_price=min(min_price,prices[i])
            left[i]=max(left[i-1],prices[i]-min_price)
        
        max_price=prices[-1]
        for i in range(n-2,-1,-1):
            max_price=max(max_price,prices[i])
            right[i]=max(right[i+1],max_price-prices[i])

        ans=0
        for i in range(n):
            ans=max(ans,left[i]+right[i])
        return ans
    