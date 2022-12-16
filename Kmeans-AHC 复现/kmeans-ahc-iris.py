# -*- coding: utf-8 -*-

# K-Means Clustering
# Importing the libraries
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from math import *

start = datetime.datetime.now()
# Importing the dataset
dataset = pd.read_csv('F:/【研一下学期课程】/算法导论/综合实验报告/复现/iris.csv')
X = dataset.iloc[:, 0:4].values
a = X.shape[0]
# print(a)

# Applying the k-means to the mall dataset
kmeans = KMeans(n_clusters = int(1.1*a**0.5), max_iter = 300, n_init = 10, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)
#result = y_kmeans.fit_predict(tfidf_matrix)
z = kmeans.cluster_centers_
# u = z.tolist()
# y = y_kmeans.tolist()
#for index, value in enumerate(y):
	#print(index, value)

# Visualizing the k-means clusters
# plt.scatter(X[:,0], X[:,1], s=3, c=y_kmeans)
# # plt.scatter(centers[:,0], s=10, c='k')
# plt.title('Results of K-means')
# plt.show()

# Classify and store clustering results
list0=[]
for j in range(len(kmeans.cluster_centers_)):
    list0.append([])
    for i in range(150):
        if y_kmeans[i] == j:
            list0[j].append(i)
            

#k=3
for k in range(2,int(sqrt(len(dataset)))+1):
    # Training the Hierarchical Clustering model on the dataset
    hc = AgglomerativeClustering(n_clusters = k, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(kmeans.cluster_centers_)
    
    
    list1=[]
    for u in range(k):
        list1.append([])
        for v in range(len(y_hc)):
            if y_hc[v] == u:
                list1[u].append(list0[v])
    
    print('类簇数为'+str(k)+'时：')
    for m in range(k):
        print('class {} includes:{}'.format(m+1, list1[m]),'\n')
    # print(list1)
    
    def flatten(a):
        for each in a:
            if not isinstance(each, list):
                yield each
            else:
                yield from flatten(each)
    
    if __name__ == "__main__":
        list3=list(flatten(list1))
        list3.sort()
        DD=dict.fromkeys(list3,)
        #print(list3)
        label=[]
        for x in range(len(list1)):
            list4=list(flatten(list1[x]))
            for y in list4:
                DD[y]=x+1
    
    Data = dataset.iloc[:, 0:4]
    Data['kind']=None
    kind=[]
    
    for i in Data.index:
        Data.iloc[i,-1]=DD[i]

    #path='C:/User/weiyijia/Desktop/算法导论期末/数据/聚类结果/iris测试输出结果/'
    Data.to_csv('iris_K_'+str(k)+'.csv')
# Visualizing the AHC clusters
# plt.scatter(X[:,0], X[:,1], s=3, c=y_hc)
# plt.title('Results of K-means-AHC')
# plt.show()
end = datetime.datetime.now()
print('running time:', end-start)
