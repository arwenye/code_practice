from typing import List, Optional

# 定义二叉树节点
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        '''利用遍历的性质'''
        if not preorder or not inorder:  # 递归终止条件
            return None
        
        root_val = preorder[0]  # 前序遍历的第一个元素是根
        root = TreeNode(root_val)

        # 找到根节点在中序遍历中的索引
        index = inorder.index(root_val)

        # 递归构造左子树
        root.left = self.buildTree(preorder[1:index+1], inorder[:index])

        # 递归构造右子树
        root.right = self.buildTree(preorder[index+1:], inorder[index+1:])

        return root
    

    def buildTree(self, preorder, inorder):
        '''哈希优化查询速度'''

        # 构建中序遍历值到索引的哈希表
        inorder_map = {val: idx for idx, val in enumerate(inorder)}
        
        # 定义递归函数，pre_start和in_start表示当前区间的开始位置
        def build(pre_start, in_start, in_end):
            if pre_start >= len(preorder) or in_start > in_end:
                return None
            
            root_val = preorder[pre_start]
            root = TreeNode(root_val)
            
            # 获取根节点在中序遍历中的位置
            root_inorder_index = inorder_map[root_val]
            
            # 递归构建左右子树
            # 左子树的前序区间是 [pre_start + 1, pre_start + left_size]
            # 右子树的前序区间是 [pre_start + left_size + 1, pre_end]
            left_size = root_inorder_index - in_start
            
            # 构建左子树
            root.left = build(pre_start + 1, in_start, root_inorder_index - 1)
            # 构建右子树
            root.right = build(pre_start + left_size + 1, root_inorder_index + 1, in_end)
            
            return root
        
        return build(0, 0, len(inorder) - 1)