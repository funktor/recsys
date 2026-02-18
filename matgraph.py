import numpy as np
from scipy.sparse import csr_matrix
import collections
import networkx as nx
import time, random
import matgraph_py

n = 100
p = 0.05
graph = nx.gnp_random_graph(n, p)
graph = nx.minimum_spanning_tree(graph)
graph = graph.edges()
print(graph)

# n = 8
# nodes = [0, 1, 2, 3, 4, 5, 6, 7]

# graph = [(0,2), (0,3), (0,4), (1,3), (1,5), (2,3), (2,5), (3,7), (4,5), (4,6), (4,7), (5,7)]
# graph = [(0,1), (0,2), (3,4), (3,5), (1,6), (2,7)]

mat = [[0]*n for _ in range(n)]

for i, j in graph:
    mat[i][j] = 1
    mat[j][i] = 1

a = np.array(mat, dtype=np.uint64)
adj = {i:[] for i in range(n)}

for i, j in graph:
    adj[i] += [j]
    adj[j] += [i]


def mat_mul_csr(a, b, n, m, p):
    cdata = []
    cindices = []
    cindptr = [0]

    res = []
    for a_i in range(len(a.indptr)-1):
        i = a_i
        s_a = a.indptr[i]
        e_a = a.indptr[i+1] if i+1 < len(a.indptr) else n

        for f in range(s_a, e_a):
            k = a.indices[f]
            u = a.data[f]

            s_b = b.indptr[k]
            e_b = b.indptr[k+1] if k+1 < len(b.indptr) else m

            for q in range(s_b, e_b):
                j = b.indices[q]
                v = b.data[q]

                if u > 0 and v > 0:
                    res += [(int(i), int(j), int(u*v))]
    
    res = sorted(res, key=lambda k: (k[0], k[1]))

    curr = (-1, -1)
    curr_i = -1
    curr_j = -1
    curr_v = 0
    h = 0

    for i, j, v in res:
        if (i, j) != curr:
            if i > curr_i and curr_i != -1:
                if curr_j != -1:
                    cindices += [curr_j]
                    cdata += [curr_v]
                    h += 1
                    
                cindptr += [h]

            if j > curr_j and curr_j != -1:
                cindices += [curr_j]
                cdata += [curr_v]
                h += 1

            curr = (i, j)
            curr_i = i
            curr_j = j
            curr_v = v
        else:
            curr_v += v
        
    if curr_j != -1:
        cindices += [curr_j]
        cdata += [curr_v]
        h += 1
    
    cindptr += [h]
    cindptr += [h]*(n+1-len(cindptr))
    
    return csr_matrix((cdata, cindices, cindptr), shape=(n,p), dtype=np.uint32)


def dist_mat_mul(a, b, n, m, p):
    cdata = []
    cindices = []
    cindptr = [0]

    res = []
    for i in range(len(a.indptr)-1):
        s_a = a.indptr[i]
        e_a = a.indptr[i+1]

        for f in range(s_a, e_a):
            k = a.indices[f]
            u = a.data[f]

            s_b = b.indptr[k]
            e_b = b.indptr[k+1] if k+1 < len(b.indptr) else m

            for q in range(s_b, e_b):
                j = b.indices[q]
                v = b.data[q]

                if int(u + v) > 0:
                    res += [(int(i), int(j), int(u + v))]

    res = sorted(res, key=lambda k: (k[0], k[1]))

    curr = (-1, -1)
    curr_i = -1
    curr_j = -1
    curr_v = 1e300
    h = 0

    for i, j, v in res:
        if (i, j) != curr:
            if i > curr_i and curr_i != -1:
                if curr_j != -1:
                    cindices += [curr_j]
                    cdata += [curr_v]
                    h += 1
                    
                cindptr += [h]

            if j > curr_j and curr_j != -1:
                cindices += [curr_j]
                cdata += [curr_v]
                h += 1

            curr = (i, j)
            curr_i = i
            curr_j = j
            curr_v = v
        else:
            curr_v = min(curr_v, v)
        
    if curr_j != -1:
        cindices += [curr_j]
        cdata += [curr_v]
        h += 1
    
    cindptr += [h]
    cindptr += [h]*(n+1-len(cindptr))
    
    return csr_matrix((cdata, cindices, cindptr), shape=(n,p), dtype=np.uint32)


def dist_mat_mul2(a, b, n, m, p):
    cdata = []
    cindices = []
    cindptr = [0]

    res = []
    for a_i in range(len(a.indptr)-1):
        i = a_i
        s_a = a.indptr[i]
        e_a = a.indptr[i+1] if i+1 < len(a.indptr) else n

        for f in range(s_a, e_a):
            k = a.indices[f]
            u = a.data[f]

            s_b = b.indptr[k]
            e_b = b.indptr[k+1] if k+1 < len(b.indptr) else m

            for q in range(s_b, e_b):
                j = b.indices[q]
                v = b.data[q]

                if u > 0 and v > 0:
                    res += [(int(i), int(j), int(u & v))]
    
    res = sorted(res, key=lambda k: (k[0], k[1]))

    curr = (-1, -1)
    curr_i = -1
    curr_j = -1
    curr_v = 0
    h = 0

    for i, j, v in res:
        if (i, j) != curr:
            if i > curr_i and curr_i != -1:
                if curr_j != -1:
                    cindices += [curr_j]
                    cdata += [curr_v]
                    h += 1
                    
                cindptr += [h]

            if j > curr_j and curr_j != -1:
                cindices += [curr_j]
                cdata += [curr_v]
                h += 1

            curr = (i, j)
            curr_i = i
            curr_j = j
            curr_v = v
        else:
            curr_v |= v
        
    if curr_j != -1:
        cindices += [curr_j]
        cdata += [curr_v]
        h += 1
    
    cindptr += [h]
    cindptr += [h]*(n+1-len(cindptr))
    
    return csr_matrix((cdata, cindices, cindptr), shape=(n,p), dtype=np.uint32)



def dist_mat_mul_max(a, b, n, m, p):
    cdata = []
    cindices = []
    cindptr = [0]

    res = []
    for a_i in range(len(a.indptr)-1):
        i = a_i
        s_a = a.indptr[i]
        e_a = a.indptr[i+1] if i+1 < len(a.indptr) else n

        for f in range(s_a, e_a):
            k = a.indices[f]
            u = a.data[f]

            s_b = b.indptr[k]
            e_b = b.indptr[k+1] if k+1 < len(b.indptr) else m

            for q in range(s_b, e_b):
                j = b.indices[q]
                v = b.data[q]

                if u > 0 and v > 0:
                    res += [(int(i), int(j), int(u + v))]
    
    res = sorted(res, key=lambda k: (k[0], k[1]))

    curr = (-1, -1)
    curr_i = -1
    curr_j = -1
    curr_v = 0
    h = 0

    for i, j, v in res:
        if (i, j) != curr:
            if i > curr_i and curr_i != -1:
                if curr_j != -1:
                    cindices += [curr_j]
                    cdata += [curr_v]
                    h += 1
                    
                cindptr += [h]

            if j > curr_j and curr_j != -1:
                cindices += [curr_j]
                cdata += [curr_v]
                h += 1

            curr = (i, j)
            curr_i = i
            curr_j = j
            curr_v = v
        else:
            curr_v = max(curr_v, v)
        
    if curr_j != -1:
        cindices += [curr_j]
        cdata += [curr_v]
        h += 1
    
    cindptr += [h]
    cindptr += [h]*(n+1-len(cindptr))
    
    return csr_matrix((cdata, cindices, cindptr), shape=(n,p), dtype=np.uint32)



def num_comps(a, n):
    visited = [0]*n
    c = 0
    for i in range(len(a.indptr)-1):
        s = a.indptr[i]
        e = a.indptr[i+1] if i+1 < len(a.indptr) else n
        flag = False
        for k in range(s, e):
            j = a.indices[k]
            if visited[j] == 0:
                flag = True
                visited[j] = 1
            else:
                break
        if flag:
            c += 1
    
    return c





def search(a, n, src, dst):
    b = csr_matrix(a[src:src+1])

    for _ in range(n):
        if b[0,dst] != 0:
            return True
        b = b.dot(a)

    return False

def search_matrix(a, n, src, dst):
    return matgraph_py.py_matsearch_simd(a, n, src, dst)

def search_graph(adj, n, src, dst):
    queue = collections.deque([src])
    visited = [0]*n
    visited[src] = 1

    while len(queue) > 0:
        node = queue.popleft()
        if node == dst:
            return True

        for b in adj[node]:
            if visited[b] == 0:
                queue.append(b)
                visited[b] = 1
    
    return False






def num_components_matrix(a, n):
    b = np.copy(a)
    np.fill_diagonal(b, 1)
    matgraph_py.py_matpaths_simd(b, n)
    return np.linalg.matrix_rank(b)


def num_components_graph(adj, n):
    visited = [0]*n
    c = 0
    for i in range(n):
        if visited[i] == 0:
            c += 1
            queue = collections.deque([i])
            while len(queue) > 0:
                node = queue.popleft()
                for j in adj[node]:
                    if visited[j] == 0:
                        queue.append(j)
                        visited[j] = 1
    
    return c




def sssp_graph(adj, n, src):
    queue = collections.deque([(src, 0)])
    dist = [n+1]*n
    visited = [0]*n
    visited[src] = 1

    while len(queue) > 0:
        node, d = queue.popleft()
        dist[node] = d

        for b in adj[node]:
            if visited[b] == 0:
                queue.append((b, d+1))
                visited[b] = 1
    
    return dist


def sssp_sparse_matrix(a, n, src):
    a[a == 0] = n+1
    np.fill_diagonal(a, 0)

    a = csr_matrix(a)
    b = csr_matrix(a[src:src+1])

    for _ in range(n):
        c = matgraph_py.py_dist_matmul_csr(b, a, 1, n, n)
        # c = dist_mat_mul(b, a, 1, n, n)
        b = b.minimum(c)

    return b.toarray()[0].tolist()


def sssp_dense_matrix(a, n, src):
    b = np.copy(a)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                b[i,j] = min(b[i,j], b[i,k] + b[k,j])

    return b[src].tolist()





def num_paths(a, n, src, dst):
    b = csr_matrix(a[src:src+1])
    c = csr_matrix(b)

    for _ in range(n):
        b = b.dot(a)
        c += b

    return c[0,dst]


    # b = np.copy(a)
    # return matgraph_py.py_mat_numpaths_par(b, n, src, dst)



def num_paths_graph_recurse(adj, n, src, dst, dp):
    if src == dst:
        return 1
    
    if dp[src] != -1:
        return dp[src]
    
    paths = 0
    for j in adj[src]:
        paths += num_paths_graph_recurse(adj, n, j, dst, dp)
    
    dp[src] = paths
    return dp[src]

def num_paths_graph(adj, n, src, dst):
    dp = [-1]*n
    return num_paths_graph_recurse(adj, n, src, dst, dp)






def has_cycle(a, n):
    a = csr_matrix(a)
    b = csr_matrix(a)
    out = csr_matrix(b)

    for _ in range(n):
        b = b.dot(a)
        out += b

    return out.diagonal().max() > 0

def has_cycle_eig(a, n):
    print(np.linalg.det(a))
    d, p = np.linalg.eig(a)
    print(d)
    d = np.diag(d)
    p_inv = np.linalg.inv(p)
    print(a)
    c = np.abs(p @ d @ p_inv)
    c[c < 1e-10] = 0
    print(c)
    f = d
    out = np.copy(a)

    for _ in range(n):
        f *= d
        g = np.abs(p @ f @ p_inv)
        g[g < 1e-10] = 0
        print(g)
        out += g.astype(dtype=np.int64)
    
    print(out)

    return np.max(np.diag(out)) > 0

def dfs_cycle(adj, n, i, visited):
    for j in adj[i]:
        if visited[j] == 0:
            visited[j] = 1
            h = dfs_cycle(adj, n, j, visited)
            visited[j] = 0
            if h:
                return True
        else:
            return True
        
    return False

def has_cycle_graph(adj, n):
    visited = [0]*n
    for i in range(n):
        visited[i] = 1
        h = dfs_cycle(adj, n, i, visited)
        visited[i] = 0
        if h:
            return True
        
    return False





def topological_sort(a, n):
    b = np.copy(a)
    b[b == 0] = -(n+1)
    np.fill_diagonal(b, 0)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                b[i,j] = max(b[i,j], b[i,k] + b[k,j])
                if i == j and b[i,j] > 0:
                    return []
        
    max_d = np.max(b, axis=0)
    return np.argsort(max_d).tolist()



    # a = csr_matrix(a)
    # b = csr_matrix(a)
    # in_degs = np.asarray(b.sum(axis=0, dtype=np.int32))[0]

    # output = []
    # for _ in range(n):
    #     c = np.where(in_degs == 0)[0]
    #     in_degs[in_degs == 0] = -1
    #     output += c.tolist()
    #     d = b[c]
    #     in_degs_1 = np.asarray(d.sum(axis=0, dtype=np.int32))[0]
    #     in_degs -= in_degs_1
    #     b[c] = 0

    # if in_degs.max() > -1:
    #     return []
    
    # return output


def topological_sort_graph(adj, n):
    in_degs = [0]*n
    for i in range(n):
        for j in adj[i]:
            in_degs[j] += 1
    
    arr = [x for x in range(n) if in_degs[x] == 0]
    res = arr
    while len(arr) > 0:
        new_arr = []
        for i in arr:
            for j in adj[i]:
                in_degs[j] -= 1
                if in_degs[j] == 0:
                    new_arr += [j]

        arr = new_arr[:]
        res += arr
    
    if sum([in_degs[i] > 0 for i in range(n)]) > 0:
        return []

    return res

    


# print(search(a, n, 0, 1))
# print(is_connected(a, n))
# print(num_components_matrix(a, n))
# print(num_components_graph(adj, n))
# a[a == 0] = n+1
# np.fill_diagonal(a, 0)
# print(a)
# print(dist_mat_mul(a, a, n, n, n))
# print(single_source_shortest_dist_unweighted(a, n, 0))
# print(num_paths(a, n, 0, 5))
# start = time.time()*1000
# print(has_cycle(a, n))
# end = time.time()*1000
# print("Duration 1 = ", end-start)

# start = time.time()*1000
# print(has_cycle_eig(a, n))
# end = time.time()*1000
# print("Duration 2 = ", end-start)

# print(has_cycle_graph(adj, n))
# start = time.time()*1000
# print(topological_sort(a, n))
# end = time.time()*1000
# print("Duration 1 = ", end-start)

# start = time.time()*1000
# print(topological_sort_graph(adj, n))
# end = time.time()*1000
# print("Duration 2 = ", end-start)
# b = a
# for i in range(n):
#     b = np.dot(a, b)

# c = b
# c[c > 0] = 1
# d = np.linalg.eigvals(c)

# print(d[np.abs(d) > 1.0e-10])

# a = np.random.randint(0, 10, (10,20))
# b = np.random.randint(0, 10, (20,10))
# a = csr_matrix(a)
# b = csr_matrix(b)
# c1 = mat_mul_csr(a, b, 10, 20, 10)
# c2 = a.dot(b)
# assert np.array_equal(c1.toarray(), c2.toarray()), "Not equal"

# m = 0
# p = 0
# q = 0
# for src in random.sample(range(n), k=20):
#     for dst in random.sample(range(n), k=20):
#         if src != dst:
#             print(src, dst)
#             start = time.time()*1000
#             x = search(a, n, src, dst)
#             end = time.time()*1000
#             print("Duration 1 = ", end-start)
#             p += (end-start)

#             start = time.time()*1000
#             y = search_matrix(a, n, src, dst)
#             end = time.time()*1000
#             print("Duration 2 = ", end-start)
#             p += (end-start)

#             start = time.time()*1000
#             z = search_graph(adj, n, src, dst)
#             end = time.time()*1000
#             print("Duration 3 = ", end-start)
#             q += (end-start)
#             m += 1
#             assert x == z and y == z, f"Mismatch !!! {x, y, z}"
#             print()

# print(p/m, q/m)



# for src in random.sample(range(n), k=min(n, 20)):
#     for dst in random.sample(range(n), k=min(n, 20)):
#         if src != dst:
#             start1 = time.time()*1000
#             x = num_paths(a, n, src, dst)
#             end1 = time.time()*1000

#             start2 = time.time()*1000
#             y = num_paths_graph(adj, n, src, dst)
#             end2 = time.time()*1000

#             if x > 0 and y > 0:
#                 print(src, dst)
#                 print("Duration 1 = ", end1-start1)
#                 print("Duration 2 = ", end2-start2)
#                 print(x, y)
#                 print()

#             assert x == y, f"Mismatch for {src}->{dst}, {x}, {y} !!!"
            




# start = time.time()*1000
# x = num_components(a, n)
# end = time.time()*1000
# print("Duration 1 = ", end-start)

# start = time.time()*1000
# y = num_components_graph(adj, n)
# end = time.time()*1000
# print("Duration 2 = ", end-start)
# assert x == y, "Mismatch !!!"
# print()


for src in random.sample(range(n), k=min(n, 20)):
    start1 = time.time()*1000
    x = sssp_sparse_matrix(a, n, src)
    end1 = time.time()*1000
    print("Duration 1 = ", end1-start1)

    start2 = time.time()*1000
    y = sssp_dense_matrix(a, n, src)
    end2 = time.time()*1000
    print("Duration 2 = ", end2-start2)

    start3 = time.time()*1000
    z = sssp_graph(adj, n, src)
    end3 = time.time()*1000
    print("Duration 3 = ", end3-start3)

    print(src)
    print(x)
    print(y)
    print(z)
    print()

    assert np.array_equal(np.array(x), np.array(z)) and np.array_equal(np.array(y), np.array(z)), "Mismatch !!!"