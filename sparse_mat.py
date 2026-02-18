import numpy as np
from scipy.sparse import csr_matrix
import matgraph_py
import time

n = 100
m = 20
p = 50

a = np.random.randint(0, 10, (n,m), dtype=np.uint64)
b = np.random.randint(0, 10, (m,p), dtype=np.uint64)

a_s = csr_matrix(a)
b_s = csr_matrix(b)

start1 = time.time()*1000
c = a_s.dot(b_s)
end1 = time.time()*1000
print(end1-start1)

start2 = time.time()*1000
d = matgraph_py.py_dist_matmul_csr(a_s, b_s, n, m, p).toarray()
end2 = time.time()*1000
print(end2-start2)

print(np.array_equal(c, d))
