from heapq import heappop, heappush
import heapq
from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:  # 如果输入为空，直接返回 None
            return None
        return self.merge(lists, 0, len(lists) - 1)  # 递归合并所有链表

    def merge(self, lists, left, right):
        """ 递归分治：合并 lists[left:right] 之间的链表 """
        if left == right:  # 递归终止条件
            return lists[left]
        mid = (left + right) // 2
        l1 = self.merge(lists, left, mid)   # 递归合并左半部分
        l2 = self.merge(lists, mid + 1, right)  # 递归合并右半部分
        return self.mergeTwoLists(l1, l2)  # 合并左右两个链表

    def mergeTwoLists(self, l1, l2):
        """ 归并两个有序链表（Leetcode 21 题） """
        dummy = ListNode(0)  # 创建一个哑节点，方便操作
        cur = dummy
        while l1 and l2:  # 归并排序合并两个链表
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 if l1 else l2  # 连接剩余的部分
        return dummy.next

    def mergeKLists2_1(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        
        ListNode.__lt__ = lambda a, b: a.val < b.val  # 让堆可以比较节点大小
        heap=[]#维护一个堆

        for i in lists:
            if i:
                heappush(heap,i)
        dummy=ListNode()
        cur=dummy

        while heap:
            node=heappop(heap)
            cur.next=node
            cur=cur.next

            if node.next:
                heappush(heap,node.next)
        return dummy.next

    def mergeKLists2_2(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        min_heap=[]
        
        cur=ListNode()
        dummy=cur
        for i in lists:
            if i:
                heapq.heappush(min_heap,(i.val,id(i),i))
        while min_heap:
            _,_,x=heapq.heappop(min_heap)
            cur.next=x
            cur=x
            if x.next:
                heapq.heappush(min_heap,(x.next.val,id(x.next),x.next))
        return dummy.next