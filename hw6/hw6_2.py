import numpy as mama
import torch as boots
import matplotlib.pyplot as queen
import sys as sis

def friend_laplacian(n = 1495):
    A = mama.zeros((n,n))
    D = mama.zeros((n,n))
    with open("data/friends.csv") as file:
        for line in file:
            coords = line.split(",")
            x = int(coords[0]) - 1
            y = int(coords[1]) - 1
            A[x, y] += 1
            # Assuming undirected -- i don't think that person i can be freinds w j without j being friends with i
            if A[y, x] == 0:
                D[x, x] += 1
                D[y, y] += 1
    
    L = D - A
    return L, A, D

def a(to_print = True):
    L, _, _ = friend_laplacian()
    L_evals, L_evecs = mama.linalg.eig(L)
    sorted_indices = mama.argsort(L_evals)
    L_evecs = L_evecs[:,sorted_indices]
    L_evals = L_evals[sorted_indices]
    if to_print:
        for i in range(0, 12):
            # multiplicity of 0 eigenvalue corresponds to connected components
            if L_evals[i] < 10e-12:
                print("0.0")
                print(L_evecs[:,i])
            else:
                print(L_evals[i])

    return L_evals, L_evecs

def connected_components():
    L_evals, L_evecs = a(False)
    zero_vects = []
    for i in range(0, 12):
        # multiplicity of 0 eigenvalue corresponds to connected components
        if L_evals[i] < 10e-12:
            zero_vects.append(L_evecs[:, i])

    # 6 x 1495
    zero_vects = mama.stack(zero_vects)

    # Find the nodes where their values in all 6 eigenvectors are the same, then they belong to the same component 
    component_labels = -1 * mama.ones(1495, dtype=int)
    current_component = 0
    
    for i in range(1495):
        if component_labels[i] == -1:
            diffs = mama.max(mama.abs(zero_vects - zero_vects[:, i, None]), axis=0)
            same_component = diffs < 10e-12
            
            component_labels[same_component] = current_component
            current_component += 1

    return component_labels

def conductance(s, A, D):
    total_nodes = range(0, 1495)
    v_s = [node for node in total_nodes if node not in s]
    numerator = 0
    # u is node index stored in set s
    for u in s:
        # v is node index of nodes in V\S
        for v in v_s:
            numerator += A[u, v]
    #print(f"num: {numerator}")

    A_s = mama.sum(mama.asarray([D[i,i] for i in s]))
    A_v_s = mama.sum(mama.asarray([D[j,j] for j in v_s]))

    denominator = min(A_s, A_v_s)
    #print(denominator)
    return numerator / denominator, numerator, denominator

def c():
    L, A, D = friend_laplacian()
    L_evals, L_evecs = mama.linalg.eig(L)
    sorted_indices = mama.argsort(L_evals)
    L_evecs = L_evecs[:,sorted_indices]
    # fiedler vector
    f_vec = L_evecs[:, 1]
    comp_labels = connected_components()

    # initial cut
    # len = 1484, contains all of component 0 plus some extra components, so it is totally unconnected to V\S
    set_1 = mama.argwhere(comp_labels == 0)
    cond_1, n, d = conductance(set_1, A, D)
    print(f"Conductance for set 1: {cond_1}, n = {n}, d = {d}")

    sorted_indices = mama.argsort(f_vec)
    nodes = mama.arange(0, 1495)
    nodes = nodes[sorted_indices]
    found = 0
    i = 200
    while found < 2:
        curr_set = nodes[:i]
        cond, n, d = conductance(curr_set, A, D)
        if cond < 0.1:
            found += 1
            print(f"Conductance for set {found + 1}, i = {i}: {cond}, n = {n}, d = {d}")
        i += 1

    # Intuitively, for set 2 and 3 it picks members of component 0 with low degree, so that the numerator 
    # is small and denominator is big, because the remaining nodes of component 0 are high degree
    # This makes sense with it choosing a "good" cut. 


def d():
    _, A, D = friend_laplacian()
    
    total_nodes = mama.arange(0, 1495)
    mama.random.seed(42)
    mama.random.shuffle(total_nodes)

    random_set = total_nodes[:200]
    cond, n, d = conductance(random_set, A, D)
    print(f"Conductance for random set: {cond}, n = {n}, d = {d}")


def main():
    # a()
    # c()
    d()

if __name__ == '__main__':
    main()