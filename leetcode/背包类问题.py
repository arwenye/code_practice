
def knapsack_01(n,W,weights,values):
    '''01背包问题，只能取一次'''
    dp=[[0]*(W+1) for _ in range(n+1)]

    for i in range(1,n+1):
        for j in range(W):
            if weights[i-1]<=j:
                dp[i][j]=max(dp[i-1][j],dp[i-1][j-weights[i-1]]+values[i-1])#在不取的情况上累积价值
            else:
                dp[i][j]=dp[i-1][j]
    return dp[n][W]


def knapsack_01_optimized(n,W,w,v):
    '''01背包问题的优化，滚动数组'''
    dp=[0]*(W+1)
    for i in range(n):
        for j in range(W,w[i]-1,-1):#不能覆盖前面的历史值
            dp[j]=max(dp[j-w[i]]+v[i],dp[j])
    return dp[-1]

def knapsack_complete(n,W,weights,values):
    '''完全背包问题,二维数组，特点是可重复取'''
    dp=[[0]*(W+1) for _ in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,W+1):
            if j >= weights[i-1]:  # 容量j能装下第i种物品（注意：物品列表是0索引，i对应weights[i-1]）
                dp[i][j] = max(
                    dp[i-1][j],  # 选择1：不选第i种物品 → 继承前i-1种物品在容量j下的最大价值
                    dp[i][j-weights[i-1]] + values[i-1]  # 选择2：选第i种物品 → 剩余容量j-weights[i-1]的最大价值 + 当前物品价值
                )
            else:
                dp[i][j] = dp[i-1][j]  # 容量不够，只能不选第i种物品
                return dp[n][W]
    return dp[n][W]

def knapsack_complete_optimize(n,W,weights,values):
    dp=[0]*(W+1)

    for i in range(n):
        for j in range(weights[i],W+1):
            dp[j]=max(dp[j],dp[j-weights[i]]+values[i])#因为与历史值无关，与更新后的值有关，可以从前往后
    return dp[-1]

def knapsack_2d(n, W1, W2, weights, volumes, values):
    '''二维背包，两个约束条件（重量W1和体积W2）'''
    # 三维DP数组：dp[i][j][k] 表示前i种物品，重量≤j，体积≤k时的最大价值
    dp = [[[0]*(W2+1) for _ in range(W1+1)] for _ in range(n+1)]
    
    for i in range(1, n+1):
        w, v, val = weights[i-1], volumes[i-1], values[i-1]
        # 倒序遍历重量（防止物品重复选取）
        for j in range(W1, w-1, -1):
            # 倒序遍历体积（防止物品重复选取）
            for k in range(W2, v-1, -1):
                # 状态转移：选或不选当前物品
                dp[i][j][k] = max(dp[i-1][j][k], dp[i-1][j-w][k-v] + val)
    
    # 所有循环结束后返回结果
    return dp[n][W1][W2]
        
def knapsack_2d_optimize(n,W1,W2,weights,volumes,values):
    '''二维背包，dp降到二维'''
    dp=[[0]*(W2+1) for _ in range(W1+1)]

    for i in range(n):  # 遍历每个物品
        w,v,val = weights[i],volumes[i],values[i]
        # 倒序遍历重量（从W1到w）
        for j in range(W1, w-1, -1):
            # 倒序遍历体积（从W2到v）
            for k in range(W2, v-1, -1):
                # 状态转移：不选当前物品 vs 选当前物品
                dp[j][k] = max(dp[j][k], dp[j-w][k-v] + val)
    return dp[W1][W2]

def grouped_knapsack(groups, W):
    '''
    分组背包问题：每组物品最多选一个，单容量约束
    groups: 物品组列表，格式为 [[(w1,v1), (w2,v2)], [(w3,v3), ...], ...]
    W: 背包最大容量
    '''
    n = len(groups)
    dp = [[0]*(W+1) for _ in range(n+1)]  # dp[i][j]：前i组，容量j时的最大价值
    
    for i in range(1, n+1):
        # 倒序遍历容量（防止同一组物品重复选取）
        for j in range(W, -1, -1):
            # 初始值：不选第i组的任何物品，继承前i-1组的结果
            dp[i][j] = dp[i-1][j]
            # 遍历第i组的所有物品（注意索引i-1，因为groups是0索引）
            for w, v in groups[i-1]:
                if j >= w:
                    # 状态转移：选当前物品 vs 不选
                    dp[i][j] = max(dp[i][j], dp[i-1][j - w] + v)
    
    return dp[n][W]

def grouped_knapsack_optimize(groups,W):
    '''分组背包，滚动数组优化'''
    dp=[0]*(W+1)

    for group in groups:
        for j in range(W,-1,-1):
            for w,v in group:
                if j>=w:
                    dp[j]=max(dp[j],dp[j-w]+v)
    return dp[-1]


def multi_knapsack_2d(n, W, weights, values, counts):
    '''多重背包，二维朴素实现'''
    dp = [[0]*(W+1) for _ in range(n+1)]  # dp[i][j]：前i种物品，容量j时的最大价值
    
    for i in range(1, n+1):
        # 当前物品的重量、价值、最大数量
        w = weights[i-1]
        v = values[i-1]
        cnt = counts[i-1]
        
        for j in range(1, W+1):  # 容量范围包含W
            dp[i][j] = dp[i-1][j]  # 初始值：不选当前物品
            
            # 遍历当前物品的可能选取数量（修正counts索引）
            for k in range(1, cnt+1):
                if k * w <= j:  # 容量足够装k个当前物品
                    # 状态转移：取选k个物品与不选的最大值
                    dp[i][j] = max(dp[i][j], dp[i-1][j - k*w] + k*v)
    
    return dp[n][W]


def multi_knapsack_1d(n,W,weights,values,counts):
    '''多重背包，一维空间优化'''
    dp=[0]*(W+1)
    for i in range(n):
        w,v,c=weights[i],values[i],counts[i]
        for j in range(W,w-1,-1):
            for k in range(1,c+1):
                if k*w<=j:
                    dp[j]=max(dp[j],dp[j-k*w]+k*v)
    return dp[-1]

def multi_knapsack_binary_1(n,W,weights,values,counts):
    '''多重背包，二进制优化,边拆边判断要不要装入背包'''
    dp=[0]*(W+1)
    for i in range(n):
        w,v,c=weights[i],values[i],counts[i]
        k=1
        while c>0:
            take=min(k,c)
            c-=take
            weight=take*w
            value=take*v
            for j in range(W,weight-1,-1):
                dp[j]=max(dp[j],dp[j-weight]+value)
            k<<=1
    return dp[-1]

def multi_knapsack_binary_2(n,W,weights,values,counts):
    '''多重背包，二进制优化,先全部拆分，再判断要不要装入背包'''
    dp=[0]*(W+1)

    items=[]
    for i in range(n):
        k=1
        w,v,c=weights[i],values[i],counts[i]
        num=c
        while num>0:
            take=min(k,num)#保证最终拆出的总数和原来一样
            items.append((take*w,take*v))
            num-=take
            k<<=1

    for weight,value in items:
        for j in range(W,weight-1,-1):
            dp[j]=max(dp[j],dp[j-weight]+value)
    return dp[-1]

def tree_knapsack(n,W,w,v,children):
    f=[[0]*(W+1) for _ in range(n+1)]

    def dfs(u):
        #优先处理自己
        for j in range(w[u],W+1):
            f[u][j]=v[u]

        #遍历每个下属
        for child in children[u]:
            dfs(child)
            for j in range(W,w[u]-1,-1):
                for k in range(j-w[u]+1):
                    f[u][j]=max(f[u][j],f[u][j-k]+f[child][k])
        root=1
        dfs(root)
        return max(f[root])
    
def tree_knapsack(n, W, w, v, children):
    # 节点编号从1开始，f[u][j]表示选择u后，容量j下的最大价值
    f = [[0]*(W+1) for _ in range(n+1)]
    
    def dfs(u):
        # 初始化：仅选择当前节点u（若容量足够）
        if w[u] <= W:
            for j in range(w[u], W+1):
                f[u][j] = v[u]
        
        # 递归处理子节点
        for child in children[u]:
            dfs(child)
            # 必须预留u的重量，才能给子节点分配容量
            for j in range(W, w[u]-1, -1):
                # 子节点可分配的最大容量 = 总容量j - u的重量（确保u已被选择）
                max_k = j - w[u]
                # 遍历子节点的容量分配（从1到max_k，0表示不选子节点，无需处理）
                for k in range(1, max_k+1):
                    # 状态转移：当前状态 vs （u的状态 + 子节点的状态）
                    f[u][j] = max(f[u][j], f[u][j - k] + f[child][k])
    
    dfs(1)
    # 返回根节点在所有容量下的最大价值
    return max(f[1][:W+1])

def tree_exclusive_knapsack(n, W, w, v, children):
    f = [[[0] * 2 for _ in range(W + 1)] for _ in range(n + 1)]

    def dfs(u):
        # ✅ 完整初始化
        for j in range(W + 1):
            f[u][j][0] = 0
            f[u][j][1] = v[u] if j >= w[u] else 0
        
        for child in children[u]:
            dfs(child)
            
            # ✅ 处理所有容量
            for j in range(W, -1, -1):
                # 选u的状态
                if j >= w[u]:
                    for k in range(j - w[u] + 1):
                        f[u][j][1] = max(f[u][j][1], f[u][j-k][1] + f[child][k][0])
                
                # 不选u的状态
                tmp = f[u][j][0]
                for k in range(j + 1):
                    tmp = max(tmp, f[u][j-k][0] + max(f[child][k][0], f[child][k][1]))
                f[u][j][0] = tmp

    dfs(1)
    return max(max(f[1][j][0], f[1][j][1]) for j in range(W + 1))