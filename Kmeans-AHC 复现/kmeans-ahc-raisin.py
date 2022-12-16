# -*- coding: utf-8 -*-
"""
Created on Sat May 14 19:16:23 2022

@author: liu
"""

# K-Means Clustering
# Importing the libraries
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

start = datetime.datetime.now()
# Importing the dataset
dataset = pd.read_excel('Raisin_Dataset.xlsx')
X = dataset.iloc[:, 0:7].values
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
list=[]
for j in range(len(kmeans.cluster_centers_)):
    list.append([])
    for i in range(900):
        if y_kmeans[i] == j:
            list[j].append(i)
            
# Training the Hierarchical Clustering model on the dataset
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(kmeans.cluster_centers_)

list1=[]
for u in range(2):
    list1.append([])
    for v in range(len(y_hc)):
        if y_hc[v] == u:
            list1[u].append(list[v])

for m in range(2):
    print('class {} includes:{}'.format(m+1, list1[m]),'\n')
# print(list1)

# Visualizing the AHC clusters
# plt.scatter(X[:,0], X[:,1], s=3, c=y_hc)
# plt.title('Results of K-means-AHC')
# plt.show()
end = datetime.datetime.now()
print('running time:', end-start)