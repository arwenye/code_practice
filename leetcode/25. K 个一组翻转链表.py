from typing import Optional
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        prev_group = dummy

        while True:
            # 检查剩余部分是否有 k 个节点
            kth = prev_group#遍历k个节点刚好停在这一组的最后一个上
            for _ in range(k):
                kth = kth.next
                if not kth:
                    return dummy.next
            
            # 反转 k 个节点
            prev, cur = kth.next, prev_group.next
            for _ in range(k):
                nxt = cur.next
                cur.next = prev
                prev = cur
                cur = nxt

            # 连接前一部分和翻转后的部分
            temp = prev_group.next
            prev_group.next = prev
            prev_group = temp  # 更新 prev_group 指针

        return dummy.next


    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # 先检查剩余长度是否足够 k 个
        cur = head
        count = 0
        while cur and count < k:
            cur = cur.next
            count += 1
        if count < k:
            return head  # 剩余不足 k 个，不翻转
        
        # 翻转 k 个节点
        prev, cur = None, head
        for _ in range(k):
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        
        # 递归翻转剩余部分
        head.next = self.reverseKGroup(cur, k)
        
        return prev  # 新的头节点
