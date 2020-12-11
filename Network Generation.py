from __future__ import division
from gurobipy import *
from math import *
from random import randint
import csv
import math
import random
import pandas as pd
import xlrd
import numpy as np
import sys
from collections import defaultdict
from gurobipy import *
import networkx as nx
import matplotlib.pyplot as plt
from random import *
#n (int) – Number of nodes
#m (int) – Number of edges to attach from a new node to existing nodes
def Scale_Free_generation(n,m):
    from Scale_Free import barabasi_albert_graph
    G= nx.barabasi_albert_graph(n,m)
    G=G.to_directed()
    plt.title('Scale_Free_generation(%s,%s)' %(n,m))
    nx.draw(G,node_size=500, alpha=0.8, node_color="red",with_labels=True)
    plt.savefig('C:/Directory/Scale_Free3.png')
    plt.show()
    Scale_free_links=list(G.edges())
    Scale_free_nodes=G.nodes()
    Scale_free_nodes=[i+1 for i in Scale_free_nodes]
    num_nodes=len(Scale_free_nodes)
    vs_Scale_free_Links=[]
    Initial_Link_Flow={}
    OD_Pair={}
    num_nodes=list(Scale_free_nodes)
    M=[[0]*len(num_nodes) for i in num_nodes]
    N=[[0]*len(num_nodes) for i in num_nodes]
    Cost={}
    for elem in Scale_free_links:
        vs_Scale_free_Links.append((elem[1],elem[0]))
    Scale_free_links=Scale_free_links+vs_Scale_free_Links
    Scale_free_links=[(i[0]+1,i[1]+1) for i in Scale_free_links]
    for elem in Scale_free_links:
    	Initial_Link_Flow[elem]=0
    	M[int(elem[0])-1][int(elem[1])-1]=1
    	Cost[(int(elem[0]),int(elem[1]))]=1
    print(Scale_free_links)
    print(Scale_free_nodes)
    print(Cost)
    for i in Scale_free_nodes:
    	for j in Scale_free_nodes:
    		if i!=j:
    			OD_Pair[i,j]=1
    nodes=Scale_free_nodes
    desarcs=list(G.edges())
    desarcs=[(x[0]+1,x[1]+1) for x in desarcs]
    desarcs=tuplelist(desarcs)
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return num_nodes,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
#############################################################################
#p (real between 0 and one) - the propability of link connection 
#num_nodes - number of nodes in a random graph
def Random_network_generation(num_nodes,p):
    G= nx.erdos_renyi_graph(num_nodes,p)
    G=G.to_directed()
    Random_links=G.edges()
    Random_nodes=G.nodes()
    num_nodes=[x+1 for x in Random_nodes]
    vs_Random_Links=[]
    Initial_Link_Flow={}
    OD_Pair={}
    Cost={}
    M=[[0]*len(num_nodes) for i in range(len(num_nodes))]
    N=[[0]*len(num_nodes) for i in range(len(num_nodes))]
    for elem in Random_links:
        vs_Random_Links.append((elem[1],elem[0]))
    Random_links=list(Random_links)+vs_Random_Links
    for elem in Random_links:
        Initial_Link_Flow[elem]=0
        M[int(elem[0])][int(elem[1])]=1
        Cost[(int(elem[0])+1,int(elem[1])+1)]=1
    for i in Random_nodes:
        for j in Random_nodes:
            if i!=j:
                OD_Pair[i+1,j+1]=1
    plt.title('Random_network_generation(%s,%s)' %(num_nodes,p))
    nx.draw(G,node_size=500, alpha=0.8, node_color="red",with_labels=True)
    filename='C:/Directory/Random%s.png'%(p)
    plt.savefig(filename)
    plt.show()
    nodes=Random_nodes
    Gp=G.to_undirected()
    desarcs=list(Gp.edges())
    desarcs=[(x[0]+1,x[1]+1) for x in desarcs]
    desarcs=tuplelist(desarcs)
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return num_nodes,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G

##############################################################################
#n (integer) - is number of nodes_or_number
#m (integer) - is number of routes
def Single_Depot(n,m):
    n_number=list(range(1,n+1))
    num_nodes=n_number
    print(n_number)
    import random
    import networkx as nx
    import matplotlib.pyplot as plt
    G=nx.Graph()
    G.add_nodes_from(n_number)
    Depot=1
    n=n-Depot
    num_paths=[]
    sum=0
    remained_node=n
    for r in range(0,m-1):
        temp=[]
        print(remained_node,remained_node-2*(m-r-1))
        upper_bound=random.randint(2,remained_node-2*(m-r-1))
        num_paths.append(upper_bound)
        remained_node=remained_node-upper_bound
    for i in num_paths:
        sum=i+sum
    num_paths.append(n-sum)
    st=2
    for i in num_paths:
        for j in range(st,st+i):
            if j==st:
                G.add_edge(Depot,j)
            if j<st+i-1:
               print(j,j+1)
               G.add_edge(j,j+1)
               G.add_edge(j+1,j)
        G.add_edge(st+i-1,Depot)
        print(st+i-1,Depot)
        G.add_edge(Depot,st+i-1)
        st=st+i
    G = G.to_directed()
    pos = nx.spring_layout(G, iterations=10000)
    plt.title('Singe_Depot(%s,%s)' %(n,m))
    nx.draw(G, pos,node_size=500, alpha=0.8, node_color="red",with_labels=True)
    Cross_links=G.edges()
    filename='C:/Directory/Singledepot%s.png'%(m)
    plt.savefig(filename)
    plt.show()
    Depot_links=G.edges()
    vs_Depot_Links=[]
    Initial_Link_Flow={}
    OD_Pair={}
    Cost={}
    M=[[0]*len(num_nodes) for i in range(len(n_number))]
    N=[[0]*len(num_nodes) for i in range(len(n_number))]
    print(len(M))
    for elem in Depot_links:
        Initial_Link_Flow[elem]=0
        M[int(elem[0])-1][int(elem[1])-1]=1
        Cost[(int(elem[0]),int(elem[1]))]=1
    for i in n_number:
        for j in n_number:
            if i!=j:
                OD_Pair[i,j]=1
    Gp=G.to_undirected()
    desarcs=list(Gp.edges())
    desarcs=tuplelist(desarcs)
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return num_nodes,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
###############################################################################
#n (integer) - number of nodes
#m (integer) - the number of nodes in one paths
def Crossing_Paths(n,m):
    import random
    n1=random.randint(3,n-3)
    n2=n-n1
    part1=list(range(1,n1+1))
    part2=list(range(n1+1,n+1))
    breakpoint=random.randint(2,n1)
    addpoint=random.randint(n1+1,n)
    import networkx as nx
    import matplotlib.pyplot as plt
    G=nx.Graph()
    part=part1+part2
    G.add_nodes_from(part)
    for i in range(0,len(part2)-1):
        G.add_edge(part2[i],part2[i+1])
        G.add_edge(part2[i+1],part2[i])
    for i in range(0,len(part1)-1):
        if part1[i]<breakpoint-1:
            G.add_edge(part1[i],part1[i+1])
        elif part1[i]==breakpoint-1:
            G.add_edge(part1[i],addpoint)
            G.add_edge(addpoint,part1[i+1])
        elif part[i]>=breakpoint:
            G.add_edge(part1[i],part1[i+1])
    G = G.to_directed()
    plt.title('Crossing_Paths(%s,%s)' %(n,m))
    pos = nx.spring_layout(G, iterations=10000)
    nx.draw(G, pos,node_size=500, alpha=0.8, node_color="red", with_labels=True)
    Cross_links=G.edges()
    filename='C:/Directory/CrossingPath.png'
    plt.savefig(filename)
    plt.show()
    Initial_Link_Flow={}
    OD_Pair={}
    Cost={}
    num_nodes=list(range(1,n+1))
    M=[[0]*len(num_nodes) for i in range(n)]
    N=[[0]*len(num_nodes) for i in range(n)]
    for elem in Cross_links:
        Initial_Link_Flow[elem]=0
        M[int(elem[0])-1][int(elem[1])-1]=1
        Cost[(int(elem[0]),int(elem[1]))]=1
    for i in part:
        for j in part:
            if i!=j:
                OD_Pair[i,j]=1
    Gp=G.to_undirected()
    desarcs=list(Gp.edges())
    desarcs=tuplelist(desarcs)
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return num_nodes,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
##############################################################################
#n number of nodes
#m number nodes in one side of the network
def matching_pairs(n,m):
    n_list=list(range(1,n+1))
    m_list=list(range(n+1,n+1+m))
    all_list=n_list+m_list
    import networkx as nx
    import matplotlib.pyplot as plt
    G=nx.Graph()
    G.add_nodes_from(all_list)
    print(G.nodes())

    for j in range(0,len(n_list)):
        for i in range(0,len(m_list)):
            G.add_edge(m_list[i],n_list[j])
            G.add_edge(n_list[j],m_list[i])
    G = G.to_directed()
    match_links=G.edges()
    print('m',match_links)
    plt.title('matching_pairs(%s,%s)' %(n,m))
    pos = nx.spring_layout(G, iterations=10000)
    nx.draw(G, pos,node_size=500, alpha=0.8, node_color="red", with_labels=True)
    filename='C:/Directory/matching_pairs.png'
    plt.savefig(filename)
    plt.show()
    num_nodes=list(range(1,n+m+1))
    Initial_Link_Flow={}
    OD_Pair={}
    M=[[0]*len(num_nodes) for i in range(n+m)]
    N=[[0]*len(num_nodes) for i in range(n+m)]
    Cost={}
    for elem in match_links:
        Initial_Link_Flow[elem]=0
        M[int(elem[0])-1][int(elem[1])-1]=1
        Cost[(int(elem[0]),int(elem[1]))]=1
    for i in all_list:
        for j in all_list:
            if i!=j:
                OD_Pair[i,j]=1
    desarcs=list(G.edges())
    desarcs=tuplelist(desarcs)
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    print('a',arcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return num_nodes,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
#############################################################################
# m (integer) - number of nodes in a ring graph
def Ring(m):
    m_link=list(range(1,m+1))
    import networkx as nx
    import matplotlib.pyplot as plt
    G=nx.Graph()
    G.add_nodes_from(m_link)
    for i in range(0,len(m_link)-1):
        G.add_edge(m_link[i],m_link[i+1])
        G.add_edge(m_link[i+1],m_link[i])

    G.add_edge(m_link[len(m_link)-1],m_link[0])
    G.add_edge(m_link[0],m_link[len(m_link)-1])
    G = G.to_directed()
    plt.title('Ring(%s)' %(m))
    pos = nx.spring_layout(G, iterations=1000)
    nx.draw(G, pos,node_size=500, alpha=0.8, node_color="red", with_labels=True)
    filename='C:/Directory/Ring.png'
    plt.savefig(filename)
    plt.show()
    Ring_links=G.edges()
    Initial_Link_Flow={}
    OD_Pair={}
    num_nodes=list(range(1,m+1))
    Cost={}
    M=[[0]*len(num_nodes) for i in range(m)]
    N=[[0]*len(num_nodes) for i in range(m)]
    for elem in Ring_links:
        Initial_Link_Flow[elem]=0
        M[int(elem[0])-1][int(elem[1])-1]=1
        Cost[(int(elem[0]),int(elem[1]))]=1
    for i in m_link:
        for j in m_link:
            if i!=j:
                OD_Pair[i,j]=1
    Gp=G.to_undirected()
    desarcs=list(Gp.edges())
    desarcs=tuplelist(desarcs)
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return num_nodes,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
##############################################################################OK
#n (integer) - number of nodes in a graph
#m (integer) - number of nodes forming the converging tail
# l (integer) - number of tails
def CentralRing(n,m,l):
    import matplotlib.pyplot as plt
    import networkx as nx
    node_list=list(range(1,n+1))
    ring_list=list(range(1,m+1))
    import networkx as nx
    import random
    import math
    G=nx.Graph()
    G.add_nodes_from(ring_list)
    print(G.nodes())
    for i in range(0,len(ring_list)-1):
        G.add_edge(ring_list[i],ring_list[i+1])
        G.add_edge(ring_list[i+1],ring_list[i])
    G.add_edge(ring_list[-1],ring_list[0])
    G.add_edge(ring_list[0],ring_list[len(ring_list)-1])
    min_tail_size=int((n-m)/l)
    if min_tail_size==0:
        raise Exception('l is greater the number of nodes you want to atsign to each tail')
    remain=(n-m) % l
    seed=[1]
    for i in range(0,l-1):
        seed.append(1+(i+1)*math.floor(m/l))
        print(1+(i+1)*math.floor(m/l))
    print(seed)
    for i in range(0,l):
        st=seed[i]
        print(seed[i])
        for j in range(m+1+i*(min_tail_size),m+1+(i+1)*(min_tail_size)):
            print(min_tail_size)
            print(m+1+i*(min_tail_size),m+1+(i+1)*(min_tail_size))
            G.add_edge(st,j)
            G.add_edge(j,st)
            print(st,j)
            st=j
        st=st+min_tail_size
        seed[i]=st

    G = G.to_directed()
    plt.title('CentralRing(%s,%s,%s):' %(n,m,l))
    pos = nx.spring_layout(G, iterations=1000)
    nx.draw(G, pos, node_size=500, alpha=0.8, node_color="red", with_labels=True)
    while remain>0:
        for i in range(0,l):
            if remain>0:
                G.add_edge(seed[i],st+1)
                G.add_edge(st+1,seed[i])
                st=st+1
                seed[i]=st
                remain=remain-1
            else:
                break
    filename='C:/Directory/CentralRing.png'
    plt.savefig(filename)
    plt.show()
    CR_links=G.edges()
    m_link=list(G.nodes())
    Initial_Link_Flow={}
    OD_Pair={}
    Cost={}
    num_nodes=list(m_link)
    M=[[0]*len(num_nodes) for i in range(0,len(num_nodes))]
    N=[[0]*len(num_nodes) for i in range(0,len(num_nodes))]
    for elem in CR_links:
        Initial_Link_Flow[elem]=0
        M[int(elem[0])-1][int(elem[1])-1]=1
        Cost[(int(elem[0]),int(elem[1]))]=1
    for i in m_link:
        for j in m_link:
            if i!=j:
                OD_Pair[i,j]=1
    Gp=G.to_undirected()
    desarcs=list(Gp.edges())
    desarcs=tuplelist(desarcs)
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return num_nodes,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
#############################################################################
#m - number of nodes in a complete graph
def Complete(m):
    m_link=list(range(1,m+1))
    import networkx as nx
    import matplotlib.pyplot as plt
    G=nx.Graph()
    G.add_nodes_from(m_link)
    for i in range(0,len(m_link)):
        for j in range(0,len(m_link)):
            if i!=j:
               G.add_edge(m_link[i],m_link[j])
    G = G.to_directed()
    Gp=G.to_undirected()
    desarcs=list(Gp.edges())
    desarcs=tuplelist(desarcs)
    print(desarcs)
    print('dd',desarcs)
    Complete_links=G.edges()
    plt.title('Complete(%s):' %(m))
    pos = nx.spring_layout(G, iterations=100)
    nx.draw(G, pos,node_size=500, alpha=0.8, node_color="red", with_labels=True)
    filename='C:/Directory/Complete.png'
    plt.savefig(filename)
    plt.show()
    Initial_Link_Flow={}
    OD_Pair={}
    Cost={}
    num_nodes=m
    M=[[0]*num_nodes for i in range(m)]
    N=[[0]*num_nodes for i in range(m)]
    for elem in Complete_links:
        Initial_Link_Flow[elem]=0
        M[int(elem[0])-1][int(elem[1])-1]=1
        Cost[(int(elem[0]),int(elem[1]))]=1
    for i in m_link:
        for j in m_link:
            if i!=j:
                OD_Pair[i,j]=1
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)

    return m_link,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
##########################################################################
#n (integer) - number of nodes in  the width of a graph
#m (integer) - number of nodes in  the hieght of a graph
def grid(n,m):
    import matplotlib.pyplot as plt
    import networkx as nx
    G = nx.grid_2d_graph(n, m)  # for example a n=4xm=4 grid
    pos = nx.spring_layout(G, iterations=100)
    G = G.to_directed()
    plt.title('grid(%s,%s)' %(n,m))
    nx.draw(G, pos, node_size=500, alpha=0.8, node_color="red", with_labels=True)
    filename='C:/Directory/grid.png'
    plt.savefig(filename)
    plt.show()
    Grid_links=list(G.edges())
    Initial_Link_Flow={}
    OD_Pair={}
    num_nodes=list(G.nodes())
    n_number=len(list(G.nodes()))
    Cost={}
    M=[[0]*len(num_nodes) for i in range(n_number)]
    N=[[0]*len(num_nodes) for i in range(n_number)]
    single_atsignment={}
    List_single_atsignment=[]
    i=1
    for n in list(G.nodes()):
        single_atsignment[n]=i
        List_single_atsignment.append(i)
        i=i+1
    single_link={}
    for elem in Grid_links:
        single_link[(elem[0],elem[1])]=[single_atsignment[elem[0]],single_atsignment[elem[1]]]
    for elem in Grid_links:
        Initial_Link_Flow[elem]=0
        M[int(single_atsignment[elem[0]])-1][int(single_atsignment[elem[1]])-1]=1
        Cost[(int(single_atsignment[elem[0]]),int(single_atsignment[elem[1]]))]=1
    for i in List_single_atsignment:
        for j in List_single_atsignment:
            if i!=j:
                OD_Pair[i,j]=1
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    desarcs=[]
    for i,j in arcs:
        if (j,i) not in desarcs:
            desarcs.append((i,j))
    desarcs=tuplelist(desarcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return List_single_atsignment,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
###############################################################################
#n (integer) - number of nodes in  the width of a graph
#m (integer) - number of nodes in  the hieght of a graph
def grid_complete(n,m):
    import matplotlib.pyplot as plt
    import networkx as nx
    G = nx.grid_2d_graph(n, m)  # 4x4 grid
    pos = nx.spring_layout(G, iterations=1000)
    G = G.to_directed()
    print(G.edges())
    List_link=list(G.nodes())
    for i in range(0,n-1):
     for j in range(0,m-1):
         G.add_edge((i,j),(i+1,j+1))
         G.add_edge((i+1,j+1),(i,j))
    for i in range(1,n):
     for j in range(0,m-1):
         G.add_edge((i,j),(i-1,j+1))
         G.add_edge((i-1,j+1),(i,j))
    G=G.to_directed()
    plt.title('grid_complete(%s,%s)' %(n,m))
    nx.draw(G, pos, node_size=500, alpha=0.8, node_color="red", with_labels=True)
    filename='C:/Directory/Complete_grid.png'
    plt.savefig(filename)
    plt.show()
    Grid_links=G.edges()
    Initial_Link_Flow={}
    OD_Pair={}
    n_number=G.nodes()
    Grid_links=G.edges()
    Initial_Link_Flow={}
    OD_Pair={}
    n_number=G.nodes()
    Cost={}
    n_number=len(list(G.nodes()))
    num_nodes=list(G.nodes())
    M=[[0]*len(num_nodes) for i in range(n_number)]
    N=[[0]*len(num_nodes) for i in range(n_number)]
    single_atsignment={}
    List_single_atsignment=[]
    i=1
    for n in list(G.nodes()):
        single_atsignment[n]=i
        List_single_atsignment.append(i)
        i=i+1
    single_link={}
    for elem in Grid_links:
        single_link[(elem[0],elem[1])]=[single_atsignment[elem[0]],single_atsignment[elem[1]]]
    for elem in Grid_links:
        Initial_Link_Flow[elem]=0
        M[int(single_atsignment[elem[0]])-1][int(single_atsignment[elem[1]])-1]=1
        Cost[(int(single_atsignment[elem[0]]),int(single_atsignment[elem[1]]))]=1
    for i in List_single_atsignment:
        for j in List_single_atsignment:
            if i!=j:
                OD_Pair[i,j]=1
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    desarcs=[]
    for i,j in arcs:
        if (j,i) not in desarcs:
            desarcs.append((i,j))
    desarcs=tuplelist(desarcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return List_single_atsignment,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
###############################################################################
# r (integer) - the num of children
# h (integer) - the height of graph
def tree(r,h):
    import matplotlib.pyplot as plt
    import networkx as nx
    G = nx.balanced_tree(r, h)
    pos = nx.spring_layout(G, iterations=100)
    G = G.to_directed()
    #pos = graphviz_layout(G, prog='twopi', args='')
    plt.title('tree(%s,%s)' %(r,h))
    nx.draw(G, pos,node_size=500, alpha=0.8, node_color="red", with_labels=True)
    plt.axis('equal')
    filename='C:/Directory/tree.png'
    plt.savefig(filename)
    plt.show()
    tree_links=G.edges()
    Initial_Link_Flow={}
    OD_Pair={}
    Cost={}
    n_number=G.nodes()
    num_nodes=list(G.nodes())
    num_nodes=[int(x)+1 for x in num_nodes]
    M=[[0]*len(num_nodes) for i in range(len(n_number))]
    N=[[0]*len(num_nodes) for i in range(len(n_number))]
    print(tree_links)
    for elem in tree_links:
        Initial_Link_Flow[elem]=0
        M[int(elem[0])][int(elem[1])]=1
        Cost[(int(elem[0])+1,int(elem[1])+1)]=1
    for i in num_nodes:
        for j in num_nodes:
            if i!=j:
                OD_Pair[i,j]=1
    G = G.to_directed()
    Gp=G.to_undirected()
    desarcs=list(Gp.edges())
    desarcs=[(x[0]+1,x[1]+1) for x in desarcs]
    desarcs=tuplelist(desarcs)
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return num_nodes,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
##################################################################################
def Hub_Spok(m):
    import networkx as nx
    import math
    import matplotlib.pyplot as plt
    #m-1 must be a multiplication of 4
    m=m-1
    plt.title('Hub_Spok(%s)' %(m))
    G=nx.Graph()
    G.add_nodes_from(range(1,m+1))
    for i in range(2,m+1):
            G.add_edge(1,i)
    G = G.to_directed()
    pos = nx.spring_layout(G, iterations=100)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos,node_size=500, alpha=0.8, node_color="red", with_labels=True)
    filename='C:/Directory/Hub_Spok.png'
    plt.savefig(filename)
    plt.show()
    Initial_Link_Flow={}
    OD_Pair={}
    Cost={}
    n_number=G.nodes()
    num_nodes=list(n_number)
    M=[[0]*len(num_nodes) for i in range(len(n_number))]
    N=[[0]*len(num_nodes) for i in range(len(n_number))]
    _links=list(G.edges())
    for elem in _links:
        Initial_Link_Flow[elem]=0
        M[int(elem[0])-1][int(elem[1])-1]=1
        Cost[(int(elem[0]),int(elem[1]))]=1
    for i in n_number:
        for j in n_number:
            if i!=j:
                OD_Pair[i,j]=1
    G = G.to_directed()
    Gp=G.to_undirected()
    desarcs=list(Gp.edges())
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return num_nodes,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
###############################################################################
#m (integer) - number of nodes in a Diamond graph --> m=4k+1 k=0,1,2,...
def Diamond(m):
    import networkx as nx
    import math
    import matplotlib.pyplot as plt
    #m-1 must be a multiplication of 4
    m=m-1
    G=nx.Graph()
    node=list(range(1,m+1))
    G.add_nodes_from(range(1,m+1))
    atsigned_node=math.floor(m/4)
    remain=m%4
    st1=1
    for i in range(2,2+atsigned_node):
            G.add_edge(st1,i)
            st1=i
    st1=1
    for i in range(2+atsigned_node,2+2*atsigned_node):
            G.add_edge(st1,i)
            st1=i
    st1=1
    for i in range(2+2*atsigned_node,2+3*atsigned_node):
            G.add_edge(st1,i)
            st1=i
    st1=1
    for i in range(2+3*atsigned_node,2+4*atsigned_node):
            G.add_edge(st1,i)
            st1=i
    print(G.edges())
    for i in range(1,atsigned_node):
        print(2+i,2+atsigned_node+i)
        print(2+atsigned_node+i,2+2*atsigned_node+i)
        print(2+2*atsigned_node+i,2+3*atsigned_node+i)
        print(2+3*atsigned_node+i,2+i)
        G.add_edge(2+i,2+atsigned_node+i)
        G.add_edge(2+atsigned_node+i,2+2*atsigned_node+i)
        G.add_edge(2+2*atsigned_node+i,2+3*atsigned_node+i)
        G.add_edge(2+3*atsigned_node+i,2+i)
    G=G.to_directed()
    pos = nx.spring_layout(G, iterations=100)
    plt.title('Diamond(%s)' %(m))
    nx.draw(G, pos,node_size=500, alpha=0.8, node_color="red", with_labels=True)
    filename='C:/Directory/Diamond.png'
    plt.savefig(filename)
    plt.show()
    Initial_Link_Flow={}
    OD_Pair={}
    Cost={}
    n_number=list(G.nodes())
    num_nodes=n_number
    M=[[0]*len(num_nodes) for i in range(len(n_number))]
    N=[[0]*len(num_nodes) for i in range(len(n_number))]
    _links=list(G.edges())
    for elem in _links:
        Initial_Link_Flow[elem]=0
        M[int(elem[0])-1][int(elem[1])-1]=1
        Cost[(int(elem[0]),int(elem[1]))]=1
    for i in n_number:
        for j in n_number:
            if i!=j:
                OD_Pair[i,j]=1
    G = G.to_directed()
    Gp=G.to_undirected()
    desarcs=list(Gp.edges())
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return num_nodes,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
################################################################################
#m (ineger) - number of nodes in a converging tail graph
def Convering_Tail(m):
    import networkx as nx
    import math
    import matplotlib.pyplot as plt
    G=nx.Graph()
    node=list(range(1,m+1))
    G.add_nodes_from(range(1,m+1))
    for i in range(1,6):
        G.add_edge(i,i+1)
        G.add_edge(i+1,i)
    G.add_edge(6,1)
    G.add_edge(1,6)
    R_m=m-6
    st1=1
    st2=4
    atsigned_node=math.floor(R_m/2)
    remain=R_m%2
    for i in range(7,7+atsigned_node):
            G.add_edge(st1,i)
            st1=i
    for i in range(7+atsigned_node,7+2*atsigned_node):
            G.add_edge(st2,i)
            st2=i
    if remain==1:
        G.add_edge(7+2*atsigned_node-1,7+2*atsigned_node)
    pos = nx.spring_layout(G, iterations=100)
    plt.figure(figsize=(8, 8))
    G=G.to_directed()
    plt.title('Convering_Tail(%s):' %(m))
    nx.draw(G, pos,node_size=500, alpha=0.8, node_color="red", with_labels=True)
    n_number=G.nodes()
    filename='C:/Diretory/Convering_Tail.png'
    plt.savefig(filename)
    plt.show()
    Initial_Link_Flow={}
    OD_Pair={}
    Cost={}
    num_nodes=G.nodes()
    n_number=G.nodes()

    M=[[0]*len(num_nodes) for i in range(len(n_number))]
    N=[[0]*len(num_nodes) for i in range(len(n_number))]
    for elem in list(G.edges()):
        Initial_Link_Flow[elem]=0
        M[int(elem[0])-1][int(elem[1])-1]=1
        Cost[(int(elem[0]),int(elem[1]))]=1
    for i in n_number:
        for j in n_number:
            if i!=j:
                OD_Pair[i,j]=1
    G = G.to_directed()
    Gp=G.to_undirected()
    desarcs=list(Gp.edges())
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return num_nodes,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
#############################################################################
#m (integer) - number of nodes in a divergin tail graph
def Diverging_Tail(m):
    import networkx as nx
    import math
    import matplotlib.pyplot as plt
    m=m-1
    branch=math.floor(m/3)
    print(branch)
    G=nx.Graph()
    node=list(range(1,m+2))
    G.add_nodes_from(node)
    remain=m%3
    st1=1
    for i in range(2,2+branch):
            G.add_edge(st1,i)
            st1=i
    st1=1
    for i in range(2+branch,2+2*branch):
            G.add_edge(st1,i)
            st1=i
    st1=1
    for i in range(2+2*branch,2+3*branch):
            G.add_edge(st1,i)
            st1=i

    if remain==1:
            G.add_edge(2+branch-1,2+3*branch)
    if remain==2:
            G.add_edge(2+branch-1,2+3*branch)
            G.add_edge(2+2*branch-1,2+3*branch+1)
    G = G.to_directed()
    DoubleU_links=G.edges()
    pos = nx.spring_layout(G, iterations=100)
    plt.title("Divergin_Tail(%s)" %(m))
    nx.draw(G, pos,node_color='red',node_size=500,alpha=0.8, with_labels=True)
    filename='C:/Directory/Divergin_Tail.png'
    plt.savefig(filename)
    plt.show()
    Initial_Link_Flow={}
    OD_Pair={}
    Cost={}
    n_number=G.nodes()
    num_nodes=G.nodes()
    M=[[0]*len(num_nodes) for i in range(len(n_number))]
    N=[[0]*len(num_nodes) for i in range(len(n_number))]
    for elem in DoubleU_links:
        Initial_Link_Flow[elem]=0
        M[int(elem[0])-1][int(elem[1])-1]=1
        Cost[(int(elem[0]),int(elem[1]))]=1
    print(M)
    print(len(M))
    for i in n_number:
        for j in n_number:
            if i!=j:
                OD_Pair[i,j]=1
    Gp=G.to_undirected()
    desarcs=list(Gp.edges())
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return num_nodes,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
##############################################################################
#m (integer) - number of nodes in a DoubleU graph
def DoubleU(m):
    import matplotlib.pyplot as plt
    import networkx as nx
    import math
    G=nx.Graph()
    node=list(range(1,m+1))
    G.add_nodes_from(range(1,m+1))
    for i in range(1,4):
        G.add_edge(i,i+1)
        G.add_edge(i+1,i)
    G.add_edge(4,1)
    G.add_edge(1,4)
    R_m=m-4
    st1=1
    st2=3
    atsigned_node=math.floor(R_m/4)
    remain=R_m%4


    for i in range(5,5+atsigned_node):
            G.add_edge(st1,i)
            st1=i
    st1=1
    for i in range(5+atsigned_node,5+2*atsigned_node):
            G.add_edge(st1,i)
            st1=i

    for i in range(5+2*atsigned_node,5+3*atsigned_node):
            G.add_edge(st2,i)
            st2=i
    st2=3
    for i in range(5+3*atsigned_node,5+4*atsigned_node):
            G.add_edge(st2,i)
            st2=i

    if remain==1:
            G.add_edge(5+atsigned_node-1,5+4*atsigned_node)
    if remain==2:
            G.add_edge(5+atsigned_node-1,5+4*atsigned_node)
            G.add_edge(5+2*atsigned_node-1,5+4*atsigned_node+1)
    if remain==3:
            G.add_edge(5+atsigned_node-1,5+4*atsigned_node)
            G.add_edge(5+2*atsigned_node-1,5+4*atsigned_node+1)
            G.add_edge(5+3*atsigned_node-1,5+4*atsigned_node+2)
    G = G.to_directed()
    DoubleU_links=G.edges()
    pos = nx.spring_layout(G, iterations=100)
    plt.title('DoubleU(%s):' %(m))
    nx.draw(G, pos, node_size=500, alpha=0.8, node_color="red", with_labels=True)
    n_number=G.nodes()
    filename='C:/Directory/DoubleU.png'
    plt.savefig(filename)
    plt.show()
    Initial_Link_Flow={}
    OD_Pair={}
    Cost={}
    n_number=G.nodes()
    num_nodes=list(n_number)
    M=[[0]*len(num_nodes) for i in range(len(n_number))]
    N=[[0]*len(num_nodes) for i in range(len(n_number))]
    for elem in DoubleU_links:
        Initial_Link_Flow[elem]=0
        M[int(elem[0])-1][int(elem[1])-1]=1
        Cost[(int(elem[0]),int(elem[1]))]=1
    for i in n_number:
        for j in n_number:
            if i!=j:
                OD_Pair[i,j]=1
    Gp=G.to_undirected()
    desarcs=list(Gp.edges())
    arcs,_Cost=multidict(Cost)
    arcs=tuplelist(arcs)
    OD_P,DS=multidict(OD_Pair)
    OD_P=tuplelist(OD_P)
    return num_nodes,arcs,desarcs,_Cost,OD_P,DS,OD_Pair,Initial_Link_Flow,M,N,Cost,G
