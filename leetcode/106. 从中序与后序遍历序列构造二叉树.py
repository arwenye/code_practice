# Definition for a binary tree node.
from typing import List,Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        '''通过标记左右子树的始末来处理'''
        inorder_map={val:idx for idx,val in enumerate(inorder)}

        def build(post_s,post_e,in_s,in_e):
            if post_e<post_s or in_s>in_e:
                return

            root_val=postorder[post_e]
            root=TreeNode(root_val)
            in_idx=inorder_map[root_val]

            root.left=build(post_s,post_s+in_idx-in_s-1,in_s,in_idx-1)
            root.right=build(post_s+in_idx-in_s,post_e-1,in_idx+1,in_e)

            return root
        
        return build(0,len(postorder)-1,0,len(inorder)-1)
    


    def buildTree(inorder, postorder):
        '''更巧妙的解法，利用根节点在最后的性质，优先构建右子树'''
        idx_map = {val: idx for idx, val in enumerate(inorder)}  # 快速定位根节点
        def helper(in_left, in_right):
            if in_left > in_right:
                return None
            root_val = postorder.pop()  # 后序最后一个是根
            root = TreeNode(root_val)
            index = idx_map[root_val]
            # 先构建右子树，再构建左子树（因为 postorder 是从右到左 pop）
            root.right = helper(index + 1, in_right)
            root.left = helper(in_left, index - 1)
            return root
        return helper(0, len(inorder) - 1)

        