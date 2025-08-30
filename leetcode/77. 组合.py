from typing import List
class Solution:
    def combine1(self, n: int, k: int) -> List[List[int]]:
        '''回溯的方法'''
        res=[]
        path=[]
        def bactracking(idx):
            if len(path)==k:
                res.append(path[:])
                return
            
            for i in range(idx,n+1):#小心，需要取到n
                path.append(i)
                bactracking(i+1)#接着i遍历，不是idx
                path.pop()
        bactracking(1)
        return res
        

    def combine2(self, n: int, k: int) -> List[List[int]]:
        '''DFS的方法'''
        res=[]
        path=[]
        def dfs(idx):
            if len(path)==k :
                res.append(path[:])
                return 
            if idx<=n:
                path.append(idx)
                dfs(idx+1)
                path.pop()
                #巧妙的分叉，选择或者不选当前这个数
                dfs(idx+1) 
            return
            
        dfs(1)
        return res