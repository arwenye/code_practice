
'''前缀和+滑动窗口'''

from typing import List
class Solution:
    def maxProfit(self, prices: List[int], strategy: List[int], k: int) -> int:
        if not prices or not strategy:
            return 0
        base=sum( s*p for p,s in zip(prices,strategy))
        half=k//2
        
        A=[(-s)*p for s,p in zip(strategy,prices)]
        B=[(1-s)*p for s,p in zip(strategy,prices)]

        sumA=sum(A[:half])
        sumB=sum(B[half:k])
        
        max_delta=max(sumA+sumB,0)
        for i in range(1,len(prices)-k+1):
            sumA=sumA-A[i-1]+A[half+i-1]
            sumB=sumB-B[half+i-1]+B[k+i-1]

            if sumA+sumB>max_delta:
                max_delta=sumA+sumB
        return base+max_delta
