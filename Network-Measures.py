#G -> Graph
#num_nodes-> list of nodes
#M -> the adjacancy list of lists of nodes
def network_measure(num_nodes,G,M):
    graph=G
    ##Count the number of Cycle 
    # Number of vertices
    V=len(graph)
    print(graph)
    def DFS(graph, marked, n, vert, start, count):

        # mark the vertex vert as visited
        marked[vert] = True

        # if the path of length (n-1) is found
        if n == 0:

            # mark vert as un-visited to make
            # it usable again.
            marked[vert] = False

            # Check if vertex vert can end with
            # vertex start
            if graph[vert][start] == 1:
                count = count + 1
                return count
            else:
                return count

        # For searching every possible path of
        # length (n-1)
        for i in range(V):
            if marked[i] == False and graph[vert][i] == 1:

                # DFS for searching path by decreasing
                # length by 1
                count = DFS(graph, marked, n-1, i, start, count)

        # marking vert as unvisited to make it
        # usable again.
        marked[vert] = False
        return count

    # Counts cycles of length
    # N in an undirected
    # and connected graph.
    def countCycles( graph, n):

        # all vertex are marked un-visited initially.
        marked = [False] * V

        # Searching for cycle by using v-n+1 vertices
        count = 0
        for i in range(V-(n-1)):
            count = DFS(graph, marked, n-1, i, i, count)

            # ith vertex is marked as visited and
            # will not be visited again.
            marked[i] = True
        return int(count)
    ###
    Conn_cycle_number=0
    ##Count the number of cycles with the length of 2 to len of the number of nodes in G
    for jj in range(2,len(M)):
        Conn_cycle_number=countCycles(M, jj)+Conn_cycle_number
    index=['Conn_cycle_number','alpha_index','Beta_index','Gamma_index','Average_degree']
    df0=pd.DataFrame(0,index=index,columns=['Results'])
    print('Conn_cycle_number',Conn_cycle_number)
    df0.at['Conn_cycle_number','Results']=Conn_cycle_number
    #Beta index: Ratio of number of links to number of nodes
    Beta_index=len(list(G.edges()))/len(list(G.nodes()))
    # Gamma index: Ratio of number of links to maximum possible number of links in a directed network
    Gamma_index=len(list(G.edges()))/(len(list(G.nodes()))(len(list(G.nodes()))-1))
    # Average Degree:The average of the length of shortest paths connecting each node to the others 
    Degree_list=G.degree(list(G.nodes()))
    Average_degree=0
    for v in Degree_list:
        Average_degree=v[1]+Average_degree
    Average_degree=Average_degree/len(list(G.nodes()))
    df0.at['Beta_index','Results']=Beta_index
    df0.at['Gamma_index','Results']=Gamma_index
    df0.at['Average_degree','Results']=Average_degree
    return Conn_cycle_number, Beta_index,Gamma_index,Average_degree
