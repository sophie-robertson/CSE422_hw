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

# Sophie's plotting
def plot_2_eigenvectors(L_graph, A_graph, n, tit):
    eigenvalues_L, eigenvectors_L = mama.linalg.eig(L_graph)
    top_2_L = boots.topk(boots.from_numpy(eigenvalues_L), k = 2).indices
    bottom_2_L = boots.topk(boots.from_numpy(eigenvalues_L), k = 2, largest=False).indices

    eigenvalues_A, eigenvectors_A = mama.linalg.eig(A_graph)
    top_2_A = boots.topk(boots.from_numpy(eigenvalues_A), k = 2).indices
    bottom_2_A = boots.topk(boots.from_numpy(eigenvalues_A), k = 2, largest=False).indices

    fig, axs = queen.subplots(2, 2)
    fig.suptitle(tit)

    axs[0, 0].scatter(range(1, n + 1), eigenvectors_L[top_2_L[0]], label = "1st Eigenvector")
    axs[0, 0].scatter(range(1, n + 1), eigenvectors_L[top_2_L[1]], label = "2nd Eigenvector")
    axs[0, 0].set_title("Laplacian Graph")
    axs[0, 0].legend(loc = "upper left")

    axs[1, 0].scatter(range(1, n + 1), eigenvectors_L[bottom_2_L[0]], label = "nth Eigenvector")
    axs[1, 0].scatter(range(1, n + 1), eigenvectors_L[bottom_2_L[1]], label = "(n-1)th Eigenvector")
    # axs[1, 0].set_title("Laplacian Graph, Smallest Eigenvectors")
    axs[1, 0].legend(loc = "upper left")

    axs[0, 1].scatter(range(1, n + 1), eigenvectors_A[top_2_A[0]], label = "nth Eigenvector")
    axs[0, 1].scatter(range(1, n + 1), eigenvectors_A[top_2_A[1]], label = "(n-1)th Eigenvector")
    axs[0, 1].set_title("Adjacency Graph")
    axs[0, 1].legend(loc = "upper left")

    axs[1, 1].scatter(range(1, n + 1), eigenvectors_A[bottom_2_A[0]], label = "nth Eigenvector")
    axs[1, 1].scatter(range(1, n + 1), eigenvectors_A[bottom_2_A[1]], label = "(n-1)th Eigenvector")
    # axs[1, 1].set_title("Adjacency Graph, Largest Eigenvectors")
    axs[1, 1].legend(loc = "upper left")

    queen.show()
    # print(eigenvectors[top_2])

def b(n = 150):
    L_cycle, A_cycle = cycle(n)
    L_spoke, A_spoke = spoke_and_wheel(n)
    L_line, A_line = line(n)
    L_lp, A_lp = line_with_point(n)
    
    plot_2_eigenvectors(L_cycle, A_cycle, n, "Cyclical Graph")

    #print(eigenvectors)
    # eigenvalues, eigenvectors = mama.linalg.eig(A_cycle)
    #print(eigenvalues)
    #print(eigenvectors)

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


def main():
    # test_graphs()
    b()

if __name__ == '__main__':
    main()