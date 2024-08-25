import numpy as np
import random
import networkx as nx

    
def partition_num(num, workers):#每线程分配任务数
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]#最后一个线程得到剩余任务


def generate_graph(ver_num):#给定顶点个数，随机连接顶点
    assert ~(isinstance(ver_num, int) and ver_num > 0),'顶点数需为正整数'
    G = nx.Graph()#初始化无向图
    H = nx.path_graph(ver_num)#
    G.add_nodes_from(H)
    def rand_edge(vi,vj,p=0.2):#随机判断是否添加一条连接两点的边
        probability =random.random()
        if(probability<p):
            G.add_edge(vi,vj)  
    i=0
    while (i<ver_num):
        j=0
        while(j<i):
            rand_edge(i,j)
            j +=1
        i +=1
    return G

    