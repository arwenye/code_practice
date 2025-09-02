# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        '''一般反转思路，三指针'''
        prev = None
        curr = head
        
        while curr:
            next_node = curr.next   # 保存当前节点的下一个节点
            curr.next = prev        # 反转当前节点的指向
            prev = curr             # 移动 prev 到当前节点
            curr = next_node        # 移动 curr 到下一个节点
            
        return prev  # 最终 prev 是反转后的链表头节点

    def reverseList(self, head: ListNode) -> ListNode:
        '''递归方法，递归后面的链表'''
        # 1. 基本情况：链表为空或只有一个节点时直接返回
        if not head or not head.next:
            return head
        
        # 2. 递归反转链表的后续部分
        new_head = self.reverseList(head.next)
        
        # 3. 反转当前节点与后续链表的连接
        head.next.next = head   # 将当前节点连接到反转后的链表后面
        head.next = None         # 断开当前节点与下一个节点的连接，防止形成环
        
        # 4. 返回新的链表头
        return new_head
