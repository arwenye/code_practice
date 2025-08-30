import heapq
from typing import List

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        '''堆排序，o(NlogN)'''
        min_heap=[]
        for i in nums:
            heapq.heappush(min_heap,nums[i])
            if len(min_heap)>k:
                heapq.heappop(min_heap)
        return min_heap[0]
    
    def findKthLargest(self, nums: List[int], k: int) -> int:
        '''快排方法，但是当序列已经有序，且pivot又选在两端，就会导致时间复杂度度变为O(n^2)'''
        def partition(left,right):
            pivot=nums[left]
            l,r=left+1,right
            while l<=r:
                while l<=r and nums[l]>=pivot:
                    l+=1
                while l<=r and nums[r]<pivot:
                    r-=1
                if l<r:
                    nums[r],nums[l]=nums[r],nums[l]
            nums[r],nums[left]=nums[r],nums[left]
            return r
        left,right=0,len(nums)-1
        k=k-1

        while left<=right:
            pos=partition(left,right)
            if pos==k:
                return nums[pos]
            if pos<k:
                left=pos+1
            else:
                right=pos-1

    def findKthLargest(self, nums: List[int], k: int) -> int:
        '''三路划分，设置数组'''
        import random
        def quick_select(numss, k):
            if not numss or len(numss) == 0:
                return -1

            pivot = random.choice(numss)
            big, equal, small = [], [], []
            
            # 1️⃣ 先遍历整个数组，划分为 big, equal, small
            for num in numss:
                if num > pivot:
                    big.append(num)
                elif num < pivot:
                    small.append(num)
                else:
                    equal.append(num)

            # 2️⃣ 再判断 k 在哪个区域
            if k <= len(big):
                return quick_select(big, k)  # 在 big 里找
            elif k > len(big) + len(equal):
                return quick_select(small, k - len(big) - len(equal))  # 在 small 里找
            else:
                return pivot  # 在 equal 里，直接返回

        return quick_select(nums, k)

    def findKthLargest3(self, nums: List[int], k: int) -> int:
        '''三路划分，原地处理'''
        import random
        def partition(nums, left, right):
            pivot_ind = random.randint(left, right)  # 随机选 pivot
            pivot = nums[pivot_ind]
            nums[pivot_ind], nums[right] = nums[right], nums[pivot_ind]  # 把 pivot 放到 right

            l, r, i = left, right - 1, left
            while i <= r:
                if nums[i] > pivot:  # 交换到左侧
                    nums[i], nums[l] = nums[l], nums[i]
                    l += 1 #l开始是和pivot相等的
                    i += 1
                elif nums[i] < pivot:  # 交换到右侧
                    nums[i], nums[r] = nums[r], nums[i]
                    r -= 1
                else:  # 等于 pivot，跳过
                    i += 1
            
            nums[right], nums[r + 1] = nums[r + 1], nums[right]  # pivot 放到正确位置
            return l, r + 1  # 返回等于 pivot 区间的左右边界
        
        def quick_select(nums, left, right, k):
            while left <= right:
                l, r = partition(nums, left, right)
                if l <= k - 1 <= r:  # k 落在等于 pivot 的范围内
                    return nums[k - 1]
                elif k - 1 < l:  # 在左边找
                    right = l - 1
                else:  # 在右边找
                    left = r + 1
        
        return quick_select(nums, 0, len(nums) - 1, k)