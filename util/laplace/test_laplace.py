import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp
import osqp
from scipy import sparse

num_sample = 128
N = num_sample * num_sample
nodes = np.arange(num_sample*num_sample)

G = nx.Graph()
G.add_nodes_from(nodes)
for i in range(0, num_sample):
    for j in range(0, num_sample):
        cur = i*num_sample+j
        if i+1 < num_sample:
            right = (i+1)*num_sample+j
            G.add_edge(cur, right)
        if j+1 < num_sample:
            down = i*num_sample+j+1
            G.add_edge(cur, down)
print("t1")
L = nx.laplacian_matrix(G)
# print(nx.laplacian_matrix(G).todense())

print("t2")

LtL = np.dot(L.transpose(), L)
# print(LtL.todense())


# fix node
rows = [0, 127, 0, 127, 63]
cols = [0, 127, 127, 0, 63]
vals = [0, 0, 0, 0, 100]

# weights node
rows_weights = [10, 30, 70]
cols_weigths = [10, 30, 70]
vals_weights = [10, 10, 10]
vals_depths = [0, 0, 0]


node_rows = []
node_cols = []

for i in range(0, len(rows)):
    node_rows.append(rows[i]*num_sample + cols[i])
    node_cols.append(rows[i]*num_sample + cols[i])
    print(rows[i]*num_sample + cols[i])

q_mat = np.zeros(num_sample*num_sample)

for i in range(0, len(rows_weights)):
    ii = rows_weights[i]*num_sample + cols_weigths[i]
    LtL[ii, ii] = LtL[ii, ii] + vals_weights[i]
    print(ii)
    q_mat[ii] = -1 * vals_weights[i] * vals_depths[i]
print("t3")

A = sparse.csc_matrix(
    ([1]*len(node_rows), (node_rows, node_cols)), shape=(N, N))
b = np.zeros(num_sample*num_sample)
for i in range(0, len(node_rows)):
    b[node_rows[i]] = vals[i]

# 节点数目
n = G.number_of_nodes()

x = cp.Variable(n)
print("t4")

n = G.number_of_nodes()

prob = osqp.OSQP()


# Setup workspace and change alpha parameter
prob.setup(LtL, q=q_mat, A=A, l=b, u=b, alpha=1.0)
res = prob.solve()
print("t5")

fout = open("node_after_optimize.txt", 'w')
n = res.x

for i in range(len(n)):
    f = n[i]
    fout.write(str(i)+" "+str(f)+"\n")
