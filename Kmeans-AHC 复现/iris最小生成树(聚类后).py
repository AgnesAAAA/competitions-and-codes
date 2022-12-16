# -*- coding: utf-8 -*-
"""
Created on Sun May 15 10:09:19 2022

@author: weiyijia
"""


import pandas as pd
# from sklearn.datasets import make_circles
# from sklearn.datasets import make_moons
# import matplotlib.pyplot as plt
import numpy as np
from math import *

#最小生成树构建

class VertexMatrix(object):
    """
    定义图中的一个顶点    
    """
    def __init__(self,data):
        self.data=data
        self.info=None


#创建图
class Graph(object):
    """
    以邻接矩阵为存储结构创建无向图
    """
    
    def __init__(self,kind):
        #图的类型：无向图，有向图，无向网，有向网
        #kind:Undigraph, Digraph, Undinetwork, Dinetwork
        self.kind=kind
        self.vertexs=[] #顶点表
        self.arcs=[]  #边表，即邻接矩阵
        self.vexnum=0 #当前顶点数
        self.arcnum=0  #当前边（弧）数
        
    def CreateGraph(self,vertex_list,edge_list):
        """

        创建图
        ----------
        vertex_list : 
            顶点列表
        edge_list : 
            边列表

        Returns
        -------        

        """
        self.vexnum=len(vertex_list)
        self.arcnum=len(edge_list)
        for vertex in vertex_list:
            vertex = VertexMatrix(vertex) 
            self.vertexs.append(vertex) #顶点列表
            self.arcs.append([float('inf')]*self.vexnum) #邻接矩阵，初始化为无穷
        for edge in edge_list:
            ivertex=self.LocateVertex(edge[0])
            jvertex=self.LocateVertex(edge[1])
            weight=edge[2]
            self.InsertArc(ivertex,jvertex,weight)
        
    def LocateVertex(self,vertex):
        """
        

        定位顶点在邻接表中的位置
        ----------
        vertex : 
            
        Returns
        -------
        
        """
        index = 0
        while index < self.vexnum:
            if self.vertexs[index].data == vertex:
                return index
            else:
                index=index+1
    
    def InsertArc(self,ivertex,jvertex,weight):
        """
        

        创建邻接矩阵
        ----------
        ivertex 
        jvertex 
        weight 
        Returns
        -------
        
        """
        if self.kind == 'Undinetwork':
            self.arcs[ivertex][jvertex] = weight
            self.arcs[jvertex][ivertex] = weight
            
    
    def GetMin(self,closedge):
        """

        Prim算法——找到当前closedge中权值最小的边
        ----------
        closedge : 

        Returns
        -------
        
        """
        index = 0
        vertex = 0
        minweight = float('inf')
        while index < self.vexnum:
            if closedge[index][1] != 0 and closedge[index][1] < minweight:
                minweight = closedge[index][1]
                vertex = index
            index = index + 1
        return vertex
    
    def Prim(self,start_vertex):
        k = self.LocateVertex(start_vertex)
        closedge = []
        arc = []
        for index in range(self.vexnum):
            closedge.append([k,self.arcs[k][index]]) #初始化
        closedge[k][1] = 0 #起始点
        index = 1
        while index < self.vexnum:
            minedge = self.GetMin(closedge) #找到与下标k相连的边
            arc.append([self.vertexs[closedge[minedge][0]].data, self.vertexs[minedge].data, closedge[minedge][1]])
            #将当前权值最小的边加入最小生成树arc
            closedge[minedge][1]=0
            i=0
            #重新选择权值最小的边
            while i < self.vexnum:
                if self.arcs[minedge][i] < closedge[i][1]:
                    closedge[i] = [minedge, self.arcs[minedge][i]]
                i = i + 1
            index = index + 1
        return arc
    
    
    #Kruskal算法
    def AddEdges(self):
        """
        
        将连通网中的边加入到列表AddEdges中
        Returns
        -------

        """
        edges = []
        i = 0
        while i < self.vexnum:
            j = 0
            while j < self.vexnum:
                if self.arcs[i][j] != float('inf'):
                    edges.append([self.vertexs[i].data, self.vertexs[j].data, self.arcs[i][j]])
                j = j + 1
            i = i+1
        #按权重从小到大进行排序
        return sorted(edges, key = lambda item:item[2])
    
    def Kruskal(self):
        edges = self.AddEdges()
        flags = []
        for index in range(self.vexnum):
            flags.append(index)
        index = 0
        while index < len(edges):
            ivertex = self.LocateVertex(edges[index][0])
            jvertex = self.LocateVertex(edges[index][1])
            if flags[ivertex] != flags[jvertex]:
                # 两个顶点不属于同一连通分量
                # 找到它们各自的连通分量的序号
                iflag=flags[ivertex]
                jflag=flags[jvertex]
                limit = 0
                while limit < self.vexnum:
                    if flags[limit] == jflag:
                        flags[limit] = iflag
                    limit = limit + 1
                index = index +1 
            else:
                edges.pop(index)
        return edges
    
 
    
#-------------------------------------------------------------
#测试数据
data_0=pd.read_csv(r"C:\Users\weiyijia\Desktop\算法导论期末\数据\原数据\iris.csv")
n=len(data_0)

for kk in range(2,int(np.sqrt(n))+1):
    #读取数据
    data_ka=pd.read_csv(r"C:\Users\weiyijia\Desktop\算法导论期末\实验报告\代码\聚类\iris聚类结果\iris_K_"+str(kk)+".csv")
    data_ka=data_ka.iloc[:,1:]
    
    #各类的边三元组，保存在字典D中，键为类别
    kind=list(data_ka['kind'].unique())
    k=len(kind)
    type_list=list(np.arange(1,len(kind)+1))
    
    List00=[]
    D=dict.fromkeys(type_list,List00)
    for k in D.keys():
        list_0=[]
        df = data_ka[data_ka['kind'] == k]
        df = df.iloc[:,0:4]
        for i in range(len(df)):
            for j in range(i+1,len(df)):
                list_0.append((df.iloc[i,:].name,df.iloc[j,:].name,np.linalg.norm(df.iloc[i,:]-df.iloc[j,:])))
                D[k]=list_0
    
    
    #各类的最小生成树，保存在字典T中，键为类别
    Tree_list=[]
    T=dict.fromkeys(type_list,Tree_list)
    
    if __name__ == '__main__':
        for m in T.keys():
            df = data_ka[data_ka['kind'] == m]
            vertex_list=list(df.index)
            edge_list=D[m]
            graph = Graph(kind='Undinetwork')
            graph.CreateGraph(vertex_list=vertex_list,
                              edge_list=edge_list)
            
            mst2 = graph.Kruskal() 
            T[m]=mst2  #各类的最小生成树，每一类的最小生成树以二维列表的形式储存，二维列表中每一个元素为边三元组
            #print('第'+str(m)+'类的Kruskal最小生成树为: ')
            # for edge in mst2:
            #     print('{0}-->{1}: {2}'.format(edge[0], edge[1], edge[2]))
                
    
    # ------------------DAS指标计算----------------------    
        
        
    #簇内紧密度
    def f_cd(T,k):
        cd = []
        for i in range(1,k+1):
            W = 0
            for j in range(len(T[i])):
                W += T[i][j][2]
            cd.append(W/len(T[i]))
        return cd
        

    #簇间分离度
    def f_sd(dic,k):#x1\x2是两个簇的位置集合
        sd=[inf]*k
        for i in range(1,k+1):
            for j in range(1,k+1):
                if i != j:
                    for m in dic[i]:
                        for n in dic[j]:
                            dist = np.linalg.norm(np.c_[m]-np.c_[n])#两个点之间的欧氏距离
                            if dist < sd[i-1]:
                                sd[i-1] = dist       
        return sd

    
    
    #聚类综合度
    def f_csd(T,dic,k):
        csd=list(range(k))
        for i in range(k):
            csd[i] = (f_sd(dic,k)[i]-f_cd(T,k))[i]/(f_sd(dic,k)[i]+f_cd(T,k)[i])
        E=np.mean(csd)
        return csd,E
    
    
    if __name__ == '__main__':
        #处理数据
        dic=dict.fromkeys(type_list,)
        for i in dic.keys():
            list_a=[]
            df = data_ka[data_ka['kind'] == i]
            df= df.iloc[:,0:4]
            for j in range(len(df)):
                list_a.append(list(df.iloc[j,:]))
            dic[i]=list_a
                
        tt=f_csd(T,dic,k)
        print('k='+str(k)+'时','E(K)=',tt[1])


          