# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
from typing import Optional
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_sum = float('-inf')#注意这里，因为当只有一个根节点且为负数时，要考虑清楚。

        def dfs(node):
            if not node:
                return 0

            # 递归计算左右子树的最大贡献值
            left = max(dfs(node.left), 0)  # 负数贡献设为0
            right = max(dfs(node.right), 0)

            # 更新全局最大路径和
            self.max_sum = max(self.max_sum, node.val + left + right)

            # 返回当前节点的最大贡献值
            return node.val + max(left, right)

        dfs(root)
        return self.max_sum