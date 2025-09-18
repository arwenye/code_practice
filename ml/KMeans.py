import numpy as np
import matplotlib.pyplot as plt
from typing import List,Tuple
import random

class KMeans:
    def __init__(self,k:int,max_iters:int=100,tol:float=1e-4):
        '''
        k-means
        '''
        self.k=k
        self.max_iters=max_iters
        self.tol=tol
        self.centroids=None
        self.labels=None


    def _initialize_centroids(self,X:np.ndarray)->np.ndarray:

        n_samples,n_features=X.shape
        centroids=np.zeros((self.k,n_features))

        centroids[0]=X[random.randint(0,n_samples-1)]

        for i in range(1,self.k):
            distances=np.array([min([np.linalg.norm(x-c)**2 for c in centroids[:i]])for x in X])
            probalities=distances/distances.sum()
            cumulative_prob=probalities.cumsum()# 计算累积概率
            r=random.random()# 生成0~1的随机数

            for j,prob in enumerate(cumulative_prob):
                if r<prob:
                    centroids[i]=X[j]
                    break
        return centroids
    
    def _assign_clusters(self,X,centroids):

        n_samples=X.shape[0]
        labels=np.zeros(n_samples,dtype=int)

        for i in range(n_samples):
            distances=[np.linalg.norm(X[i]-c) for c in centroids]
            labels[i]=np.argmin(distances)
        
        return labels
    
    def _update_centroids(self,X,labels):

        n_features=X.shape[1]
        centroids=np.zeros((self.k,n_features))

        for i in range(self.k):
            cluster_points=X[labels==i]
            if len(cluster_points)>0:
                centroids[i]=cluster_points.mean(axis=0)
            else:
                centroids[i]=self.centroids[i]
        return centroids
    
    def _calculate_cost(self,X,labels,centroids):
        cost=0
        for i in range(self.k):
            cluster_points=X[labels==i]
            if len(cluster_points)>0:
                cost+=np.sum((cluster_points-centroids[i])**2)
        return cost
    def fit(self,X):
        self.centroids=self._initialize_centroids(X)
        prev_cost=float('inf') 

        for i in range(self.max_iters):
            labels=self._assign_clusters(X,self.centroids)
            new_centroids=self._update_centroids(X,labels)
            current_cost=self._calculate_cost(X,labels,new_centroids)
            if abs(prev_cost-current_cost)<self.tol:
                print('')
                break

            self.centroids=new_centroids
            prev_cost=current_cost
        self.labels=labels
        return self
    
    def predict(self,X):
        return self._assign_clusters(X,self.centroids)
    
    def fit_predict(self,X):
        self.fit(X)
        return self.labels
    
def generate_sample_data(n_samples: int = 300) -> np.ndarray:
    """
    生成示例数据用于测试
    """
    # 创建3个高斯分布的簇
    cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(n_samples//3, 2))
    cluster2 = np.random.normal(loc=[6, 6], scale=0.5, size=(n_samples//3, 2))
    cluster3 = np.random.normal(loc=[2, 6], scale=0.5, size=(n_samples//3, 2))
    
    return np.vstack([cluster1, cluster2, cluster3])

def plot_clusters(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray, title: str):
    """
    可视化聚类结果
    """
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    for i in range(len(np.unique(labels))):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[i], alpha=0.6, label=f'Cluster {i+1}')
    
    # 绘制聚类中心
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               c='black', marker='x', s=200, linewidths=3, label='Centroids')
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    X = generate_sample_data(300)
    
    # 创建K-means模型
    kmeans = KMeans(k=3, max_iters=100)
    
    # 训练模型
    labels = kmeans.fit_predict(X)
    
    # 输出结果
    print(f"聚类中心:")
    for i, centroid in enumerate(kmeans.centroids):
        print(f"Cluster {i+1}: ({centroid[0]:.2f}, {centroid[1]:.2f})")
    
    print(f"\n各簇的样本数量:")
    unique, counts = np.unique(labels, return_counts=True)
    for i, count in enumerate(counts):
        print(f"Cluster {i+1}: {count} 个样本")
    
    # 可视化结果
    plot_clusters(X, labels, kmeans.centroids, "K-means 聚类结果")
    
    # 测试预测新数据点
    new_points = np.array([[1, 1], [5, 5], [3, 7]])
    predictions = kmeans.predict(new_points)
    print(f"\n新数据点的聚类预测:")
    for i, (point, pred) in enumerate(zip(new_points, predictions)):
        print(f"点 ({point[0]}, {point[1]}) 属于 Cluster {pred + 1}")