
def lengthOfLongestSubstring(self, s: str) -> int:
    '''使用散列字典来标记位置，如果相同且在左边界里面就重复，要更新'''
    app={}
    n=len(s)
    ans=0
    l=0
    for i in range(n):
        if s[i] not in app:
            app[s[i]]=i
        elif app[s[i]]<l:
            app[s[i]]=i
        else:
            ans=max(ans,i-l)
            l=app[s[i]]+1
            app[s[i]]=i
    ans=max(ans,n-l)
    return ans

def lengthOfLongestSubstring(self, s: str) -> int:
        '''使用集合，维持一个滑动窗口，有重复就移动到没有重复为止'''
        app=set()
        n=len(s)
        ans=0
        l=0
        for i in range(n):
            if s[i] not in app:
                app.add(s[i])
            else:
                ans=max(ans,len(app))
                while s[i] in app:
                    app.remove(s[l])
                    l+=1
                app.add(s[i])
        ans = max(ans, len(app))
        return ans

