class Solution:
    def longestPalindrome1(self, s: str) -> str:
        '''动态规划解法'''
        n = len(s)
        if n < 2:
            return s  # 长度小于2，直接返回本身
        
        dp = [[False] * n for _ in range(n)]
        start, max_len = 0, 1  # 记录最长回文子串的起点和长度
        
        for j in range(n):  # 遍历右边界
            for i in range(j + 1):  # 遍历左边界
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    if j - i + 1 > max_len:  # 记录最长回文子串的信息
                        start, max_len = i, j - i + 1
        
        return s[start:start + max_len]
    
    def longestPalindrome2_1(self, s: str) -> str:
        '''中心扩展法,使用左右边界来标识'''
        def expand_around_center(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return left + 1, right - 1  # 返回回文的左右边界

        start, end = 0, 0  # 记录最长回文的起始位置
        for i in range(len(s)):
            # 单个字符为中心
            l1, r1 = expand_around_center(i, i)
            # 两个相邻字符为中心
            l2, r2 = expand_around_center(i, i + 1)

            # 更新最长回文子串
            if r1 - l1 > end - start:
                start, end = l1, r1
            if r2 - l2 > end - start:
                start, end = l2, r2

        return s[start:end + 1]
        
    def longestPalindrome2_2(self, s: str) -> str:
        '''中心扩展法,使用左边界和长度来标识'''
        def expand_around_center(left,right):
            while left>=0 and right<len(s) and s[left]==s[right]:
                left-=1
                right+=1
            return left+1,right-1
        
        start,max_len=0,1
        for i in range(len(s)):
            l1,r1=expand_around_center(i,i)
            l2,r2=expand_around_center(i,i+1)
            if r1-l1+1>max_len:
                start,max_len=l1,r1-l1+1
            if r2-l2+1>max_len:
                start,max_len=l2,r2-l2+1
        
        return s[start:start+max_len]
    def longestPalindrome(self, s: str) -> str:
        '''Manacher解法，O(n)时间复杂度，优化的关键在于通过对称性来减少回文计算的次数，因为维护了最后边的回文串，每次可以在对称点或者有边界的基础上继续拓展'''
        # 1. 预处理字符串
        t = "#" + "#".join(s) + "#"  # 变为奇数长度
        n = len(t)  # 预处理后的长度
        p = [0] * n  # 记录回文半径
        C, R = 0, 0  # 记录当前回文的中心C，最右端R
        max_len, center_index = 0, 0  # 记录最长回文的信息

        # 2. 遍历预处理后的字符串
        for i in range(n):
            # 计算 p[i] 初始值
            mirror = 2 * C - i  # i 关于 C 的对称点 i'
            if i < R:
                p[i] = min(R - i, p[mirror])  # 不超过 R 的部分直接赋值

            # 尝试中心扩展
            while i + p[i] + 1 < n and i - p[i] - 1 >= 0 and t[i + p[i] + 1] == t[i - p[i] - 1]:
                p[i] += 1
            
            # 更新 C 和 R
            if i + p[i] > R:
                C, R = i, i + p[i]

            # 记录最长回文信息
            if p[i] > max_len:
                max_len, center_index = p[i], i

        # 3. 还原原字符串的最长回文子串
        start = (center_index - max_len) // 2  # 计算原始索引
        return s[start:start + max_len]

        
