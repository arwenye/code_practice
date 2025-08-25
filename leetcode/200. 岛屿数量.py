from collections import deque
from typing import List


class Solution:
    def numIslands1(self, grid: List[List[str]]) -> int:
        '''DFS方法'''
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        count = 0

        def dfs(r, c):
            # 越界 或 遇到水（0） 直接返回
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
                return
            
            # 标记为已访问
            grid[r][c] = '0'
            
            # 递归访问上下左右
            dfs(r - 1, c)  # 上
            dfs(r + 1, c)  # 下
            dfs(r, c - 1)  # 左
            dfs(r, c + 1)  # 右

        # 遍历整个网格
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':  # 遇到陆地，开始 DFS
                    count += 1
                    dfs(r, c)
        
        return count

    def numIslands2(self, grid: List[List[str]]) -> int:
        '''BFS'''
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        count = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右方向

        def bfs(r, c):
            queue = deque([(r, c)])
            grid[r][c] = '0'  # 标记为已访问

            while queue:
                x, y = queue.popleft()
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == '1':
                        grid[nx][ny] = '0'
                        queue.append((nx, ny))

        # 遍历整个网格
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':  # 遇到陆地，开始 BFS
                    count += 1
                    bfs(r, c)
        
        return count
    

    def numIslands3(self, grid: List[List[str]]) -> int:
        '''并查集'''
        if not grid or not grid[0]:  # 边界情况
            return 0
        
        uf = UnionFind(grid)  # 创建并查集
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 方向：上、下、左、右
        rows, cols = len(grid), len(grid[0])

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':  # 遇到陆地
                    grid[r][c] = '0'  # 标记访问
                    for dr, dc in directions:  # 遍历四个方向
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                            uf.union((r, c), (nr, nc))  # 合并相邻陆地

        return uf.count  # 返回岛屿数量


class UnionFind:
    def __init__(self, grid):
        self.parent = {}  # 记录每个节点的根
        self.rank = {}    # 记录树的高度（路径压缩优化）
        self.count = 0    # 记录连通分量（岛屿数）

        # 初始化并查集
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == '1':  # 只有陆地才是节点
                    self.parent[(r, c)] = (r, c)  # 初始时每个节点指向自己
                    self.rank[(r, c)] = 0  # 初始高度为0
                    self.count += 1  # 每个 '1' 先看作独立的岛屿

    def find(self, node):
        # 查找根节点，并进行路径压缩
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])  # 递归找根，并直接指向根
        return self.parent[node]

    def union(self, node1, node2):
        root1, root2 = self.find(node1), self.find(node2)
        if root1 != root2:  # 只有不同的根节点才需要合并
            # 按秩合并（优化）
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1  # 把 root2 挂到 root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2  # 把 root1 挂到 root2
            else:
                self.parent[root2] = root1  # 挂任意一个，并增加高度
                self.rank[root1] += 1
            self.count -= 1  # 合并成功后，岛屿数减少

