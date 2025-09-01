class Solution:
    def longestConsecutive(self, nums):
        '''使用集合，哈希的方式。连续只用验证下一个在不在里面，为了不重复，要求上一个不在才开始。'''
        num_set = set(nums)
        longest = 0

        for num in num_set:
            if num - 1 not in num_set:  # 只从起点开始
                current = num
                length = 1
                while current + 1 in num_set:
                    current += 1
                    length += 1
                longest = max(longest, length)

        return longest
