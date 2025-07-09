from collections import deque

# This is the function that you will implement the greedy algorithm
# It should input a graph as an adjacency list and return a partition where 
# every vertex is balanced, as described above.
# Return a list of Booleans of size n, wherein for any vertex v, if list[v] = True then 
# it belongs to partition S1 otherwise to partition S2.
def find_balanced_cut(n, adj_list): 
    assert n >= 1
    assert len(adj_list) == n
    # Check that the adjacency list makes sense and represents an undirected graph
    for (i, neighbors) in enumerate(adj_list):
        assert all( 0 <= j < n for j in neighbors )
        assert i not in neighbors # no self loops allowed
        for j in neighbors: 
            assert i in adj_list[j]
            
    # Start with an initial cut that places first n/2 nodes in S1 and rest in S2.
    cut = [True if i < n//2 else False for i in range(n)]
    ## TODO: now run the greedy algorithm. It will be helpful to have helper functions to find 
    ## imbalanced_vertices, maintain an array with the number of edges for each node that are cut and so on.
    ## Note: your algorithm must return a cut where all nodes are balanced.
    #my code here
    degrees = [len(neighbors) for neighbors in adj_list]
    
    # Maintain a count of internal edges for each vertex.
    internal_counts = [0] * n
    for i in range(n):
        count = 0
        for neighbor in adj_list[i]:
            if cut[i] == cut[neighbor]:
                count += 1
        internal_counts[i] = count
        
    # Use a queue to keep track of imbalanced vertices that need processing.
    imbalanced_queue = deque()
    for i in range(n):
        if 2 * internal_counts[i] > degrees[i]:
            imbalanced_queue.append(i)
            
    # --- Greedy Algorithm Main Loop ---
    # Continue as long as there are potentially imbalanced vertices to process.
    while imbalanced_queue:
        v = imbalanced_queue.popleft()
        if 2 * internal_counts[v] <= degrees[v]:
            continue
        v_old_partition_is_true = cut[v]
        cut[v] = not v_old_partition_is_true
        internal_counts[v] = degrees[v] - internal_counts[v]
        
        for u in adj_list[v]:

            if cut[u] == v_old_partition_is_true:
                internal_counts[u] -= 1

            else:
                internal_counts[u] += 1

            if 2 * internal_counts[u] > degrees[u]:
                imbalanced_queue.append(u)

    return cut


#These  are useful functions for the test cases
# IMPORTANT: 
# Please ensure that you run these cells before running test cases or else you may get unknown function errors.

# Make an adjacency list out of a list of edges.
def mk_adjacency_list(n, edge_list):
    adj_list = [set() for i in range(n)]
    for (i,j) in edge_list:
        adj_list[i].add(j)
        adj_list[j].add(i)
    return adj_list

# Test Partition
def test_cut(n, adj_list, cut):
    num_edges_crossing_cut = [0]*n
    for (i, neighbors) in enumerate(adj_list):
        num_edges_crossing_cut[i] = sum([cut[i] != cut[j] for j in neighbors])
        if 2 * num_edges_crossing_cut[i] < len(neighbors):
            assert False, f'Test Failed: In your cut, vertex {i} has {len(neighbors)} edges incident on it but only {num_edges_crossing_cut[i]} edges cross the cut'
    return 
    
## WARNING: these graphs are going to be large. Make sure that your code is efficient enough to finish running this cell
## within 1 minute at worst.
## Our referene solution finishes in nearly 100 milli seconds on a macbook pro laptop
# running intel core i7 3.1 GHz processor
# if you are curious.
from random import randint
def mk_random_graph(n, m):
    adj_list = [set() for i in range(n)]
    for k in range(m):
        i = randint(0, n-1)
        j = randint(0, n-1)
        if i == j: 
            continue
        adj_list[i].add(j)
        adj_list[j].add(i)
    return adj_list


adj_list = mk_random_graph(100, 1000) # making random graph with 100 nodes and 10000 edges
cut = find_balanced_cut(100, adj_list)
test_cut(100, adj_list, cut)


adj_list = mk_random_graph(100, 1000) # making random graph with 100 nodes and 1000 edges
cut = find_balanced_cut(100, adj_list)
test_cut(100, adj_list, cut)


adj_list = mk_random_graph(250, 2500) # making random graph with 250 nodes and 2500 edges
cut = find_balanced_cut(250, adj_list)
test_cut(250, adj_list, cut)



adj_list = mk_random_graph(500, 10000) # making random graph with 250 nodes and 2500 edges
cut = find_balanced_cut(500, adj_list)
test_cut(500, adj_list, cut)

print('Test Passed (15 points)')
