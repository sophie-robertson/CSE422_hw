import numpy as mama
import torch as boots
import matplotlib.pyplot as queen

def cycle(n):
    D = 2 * mama.eye(n)
    A = mama.zeros((n,n))
    for i in range(n):
        A[i][(i+1) % n] = 1
        A[(i+1) % n][i] = 1
    L = D - A
    return L, A

def spoke_and_wheel(n):
    L = mama.zeros((n,n))
    L_cycle, _ = cycle(n - 1)
    L[0:-1,0:-1] = L_cycle + mama.eye(n-1)
    for i in range(n):
        L[-1][i] = -1
        L[i][-1] = -1
    L[-1][-1] = n - 1
    A = mama.diag(mama.diag(L)) - L
    return L, A

def line(n):
    L_cycle, A_cycle = cycle(n)
    L = L_cycle
    L[0][0] = 1
    L[-1][-1] = 1
    L[-1][0] = 0
    L[0][-1] = 0
    A = A_cycle
    A[-1][0] = 0
    A[0][-1] = 0
    return L, A

def line_with_point(n):
    L = mama.zeros((n,n))
    L_line, _ = line(n - 1)
    L[0:-1,0:-1] = L_line + mama.eye(n-1)
    for i in range(n):
        L[-1][i] = -1
        L[i][-1] = -1
    L[-1][-1] = n - 1
    A = mama.diag(mama.diag(L)) - L
    return L, A

def test_graphs():
    L_cycle, A_cycle = cycle(7)
    L_spoke, A_spoke = spoke_and_wheel(7)
    L_line, A_line = line(7)
    L_lp, A_lp = line_with_point(7)
    print('Laplacians')
    print(L_cycle)
    print(L_spoke)
    print(L_line)
    print(L_lp)
    print()
    print('Adjacency')
    print(A_cycle)
    print(A_spoke)
    print(A_line)
    print(A_lp)

# Sam's plotting
def plot_eigenvectors(L, A, graph):
    L_evals, L_evecs = mama.linalg.eig(L)
    # Ensure eigenvalues/eigenvectors are sorted
    sorted_indices = mama.argsort(L_evals)
    L_evecs = L_evecs[:,sorted_indices]
    L_evals = L_evals[sorted_indices]

    A_evals, A_evecs = mama.linalg.eig(A)
    # Ensure eigenvalues/eigenvectors are sorted
    sorted_indices = mama.argsort(A_evals)
    A_evecs = A_evecs[:,sorted_indices]
    A_evals = A_evals[sorted_indices]

    fig, axs = queen.subplots(2, 2)
    fig.subplots_adjust(hspace=0.5)
    x = mama.arange(A_evals.size) + 1

    # Smallest laplacian eigenvectors
    axs[0,0].scatter(x, L_evecs[:,0], s=0.1, c='red', label=f'Smallest')
    axs[0,0].scatter(x, L_evecs[:,1], s=0.1, c='blue', label=f'Second Smallest')
    axs[0,0].set_xlabel('Indices')
    axs[0,0].set_ylabel('Eigenvectors')
    axs[0,0].set_title(f'Smallest Laplacian Eigenvectors for {graph} Graph')
    axs[0,0].legend()

    # Largest laplacian eigenvectors
    axs[0,1].scatter(x, L_evecs[:,-1], s=0.1, c='red', label=f'Largest')
    axs[0,1].scatter(x, L_evecs[:,-2], s=0.1, c='blue', label=f'Second Largest')
    axs[0,1].set_xlabel('Indices')
    axs[0,1].set_ylabel('Eigenvectors')
    axs[0,1].set_title(f'Largest Laplacian Eigenvectors for {graph} Graph')
    axs[0,1].legend()

    # Smallest adjacency eigenvectors
    axs[1,0].scatter(x, A_evecs[:,0], s=0.1, c='red', label='Smallest')
    axs[1,0].scatter(x, A_evecs[:,1], s=0.1, c='blue', label='Second Smallest')
    axs[1,0].set_xlabel('Indices')
    axs[1,0].set_ylabel('Eigenvectors')
    axs[1,0].set_title(f'Smallest Adjacency Eigenvectors for {graph} Graph')
    axs[1,0].legend()

    # Largest adjecency eigenvectors
    axs[1,1].scatter(x, A_evecs[:,-1], s=0.1, c='red', label='Largest')
    axs[1,1].scatter(x, A_evecs[:,-2], s=0.1, c='blue', label='Second Largest')
    axs[1,1].set_xlabel('Indices')
    axs[1,1].set_ylabel('Eigenvectors')
    axs[1,1].set_title(f'Largest Adjacency Eigenvectors for {graph} Graph')
    axs[1,1].legend()


    print(graph)
    print("Laplacian Eigenvalues")
    print(L_evals[0], L_evals[1], L_evals[-2], L_evals[-1])
    print('Adjacency Eigenvalues')
    print(A_evals[0], A_evals[1], A_evals[-2], A_evals[-1])
    print()
    # (L_evals[:,0], L_evals[:,1], L_evals[:,-2], L_evals[:,-1]), (A_evals[:,0], A_evals[:,1], A_evals[:,-2], A_evals[:,-1])

    #queen.tight_layout()
    queen.show()
    # queen.savefig(f'images/{graph}.png')

def b():
    n = 150
    L_cycle, A_cycle = cycle(n)
    L_spoke, A_spoke = spoke_and_wheel(n)
    L_line, A_line = line(n)
    L_lp, A_lp = line_with_point(n)

    plot_eigenvectors(L_cycle, A_cycle, "Cycle")
    plot_eigenvectors(L_spoke, A_spoke, "Spoke and Wheel")
    plot_eigenvectors(L_line, A_line, "Line")
    plot_eigenvectors(L_lp, A_lp, "Line and Point")

def e():
    for n in [5, 10, 100]:
        print(f'n = {n}')
        print()
        L_cycle, A_cycle = cycle(n)
        L_spoke, A_spoke = spoke_and_wheel(n)
        L_line, A_line = line(n)
        L_lp, A_lp = line_with_point(n)

        spoke = n**2 - 3*n + 4
        l = 4
        lp = n**2 - 3*n

        print('Spoke and wheel')
        print(f'Calculated: {spoke}')
        print(f'Actual: {mama.linalg.norm(L_cycle - L_spoke) ** 2}')
        print()

        print('Line')
        print(f'Calculated: {l}')
        print(f'Actual: {mama.linalg.norm(L_cycle - L_line) ** 2}')
        print()

        print('Line with point')
        print(f'Calculated: {lp}')
        print(f'Actual: {mama.linalg.norm(L_cycle - L_lp) ** 2}')
        print()

        # assert mama.linalg.norm(L_cycle - L_spoke) ** 2 == spoke
        # assert mama.linalg.norm(L_cycle - L_line) ** 2 == l
        # assert mama.linalg.norm(L_cycle - L_lp) ** 2 == lp

def f():
    # Generate graph
    n = 600
    points = 2 * mama.random.rand(n, 2)
    D = mama.zeros(n)
    A = mama.zeros((n,n))
    for i in range(n):
        for j in range(i + 1, n):
            if mama.linalg.norm(points[i,:] - points[j,:]) <= 0.5:
                # Add edges
                A[i][j] = 1
                A[j][i] = 1
                # Add degrees
                D[i] += 1
                D[j] += 1
    L = mama.diag(D) - A

    # Find points with x,y < 0
    max_row = mama.max(points, axis = 1)
    small_indices = mama.nonzero(mama.where(max_row < 1, 1, 0))
    big_indices = mama.nonzero(mama.where(max_row >= 1, 1, 0))

    # Calculate eigenvectors
    L_evals, L_evecs = mama.linalg.eig(L)
    # Ensure eigenvalues/eigenvectors are sorted
    sorted_indices = mama.argsort(L_evals)
    L_evecs = L_evecs[:,sorted_indices]
    L_evals = L_evals[sorted_indices]

    queen.scatter(L_evecs[:,1][small_indices], L_evecs[:,2][small_indices], s=2, c='red', label='x,y < 1')
    queen.scatter(L_evecs[:,1][big_indices], L_evecs[:,2][big_indices], s=2, c='blue', label='x >= 1 or y >= 1')
    queen.xlabel('Second Smallest Eigenvector')
    queen.ylabel('Third Smallest Eigenvector')
    queen.legend()
    queen.show()

def g():
    n = 100

    # Calculate grid laplacian
    A = mama.zeros((n**2, n**2))
    for index in range(n**2):
        i = index // n
        j = index % n
        min_i = max(i-1, 0)
        max_i = min(i+1, n-1)
        min_j = max(j-1, 0)
        max_j = min(j+1, n-1)
        for i_0 in range(min_i, max_i + 1):
            for j_0 in range(min_j, max_j + 1):
                A[index][n*i_0 + j_0] = 1
    D = mama.sum(A, axis=0)
    L = mama.diag(D) - A # automatically removes self-loops

    # Calculate eigenvectors
    L_evals, L_evecs = mama.linalg.eig(L)
    # Ensure eigenvalues/eigenvectors are sorted
    sorted_indices = mama.argsort(L_evals)
    L_evecs = L_evecs[:,sorted_indices]
    L_evals = L_evals[sorted_indices]
    edges = mama.nonzero(A)

    # Plot grid embedding
    queen.scatter(L_evecs[:,1], L_evecs[:,2], s=5, c='blue', zorder=2)
    for entry in edges:
        queen.plot(L_evecs[:,1][entry], L_evecs[:,2][entry], c='red', linewidth=0.2, zorder=1)
    queen.xlabel('Second Smallest Eigenvector')
    queen.ylabel('Third Smallest Eigenvector')
    queen.title('Spectral Embedding for Grid Plot')
    queen.show()

    # Remove 100 points at random
    removed = mama.random.choice(mama.arange(n**2), n)
    keep = mama.delete(mama.arange(n**2), removed)
    A_removed = (A[keep,:])[:,keep] # removed rows and columns
    D_removed = mama.sum(A_removed, axis=0)
    L_removed = mama.diag(D_removed) - A_removed

    # Calculate eigenvectors
    L_evals, L_evecs = mama.linalg.eig(L_removed)
    # Ensure eigenvalues/eigenvectors are sorted
    sorted_indices = mama.argsort(L_evals)
    L_evecs = L_evecs[:,sorted_indices]
    L_evals = L_evals[sorted_indices]
    edges = mama.nonzero(A_removed)

    # Plot removed embedding
    queen.scatter(L_evecs[:,1], L_evecs[:,2], s=5, c='blue', zorder=2)
    for entry in edges:
        queen.plot(L_evecs[:,1][entry], L_evecs[:,2][entry], c='red', linewidth=0.2, zorder=1)
    queen.xlabel('Second Smallest Eigenvector')
    queen.ylabel('Third Smallest Eigenvector')
    queen.title('Spectral Embedding with Removed Points')
    queen.show()

def main():
    # test_graphs()
    # b()
    # e()
    # f()
    g()

if __name__ == '__main__':
    main()