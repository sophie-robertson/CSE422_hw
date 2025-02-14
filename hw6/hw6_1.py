import numpy as np
import matplotlib.pyplot as plt

def cycle(n):
    D = 2 * np.eye(n)
    A = np.zeros((n,n))
    for i in range(n):
        A[i][(i+1) % n] = -1
        A[(i+1) % n][i] = -1
    L = D + A
    return L, A

def spoke_and_wheel(n):
    L = np.zeros((n,n))
    L_cycle, _ = cycle(n - 1)
    L[0:-1,0:-1] = L_cycle + np.eye(n-1)
    for i in range(n):
        L[-1][i] = -1
        L[i][-1] = -1
    L[-1][-1] = n - 1
    A = L - np.diag(np.diag(L))
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
    L = np.zeros((n,n))
    L_line, _ = line(n - 1)
    L[0:-1,0:-1] = L_line + np.eye(n-1)
    for i in range(n):
        L[-1][i] = -1
        L[i][-1] = -1
    L[-1][-1] = n - 1
    A = L - np.diag(np.diag(L))
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

def plot_eigenvectors(L, A, graph):
    L_evals, L_evecs = np.linalg.eig(L)
    # Ensure eigenvalues/eigenvectors are sorted
    sorted_indices = np.argsort(L_evals)
    L_evecs = L_evecs[:,sorted_indices]

    A_evals, A_evecs = np.linalg.eig(A)
    # Ensure eigenvalues/eigenvectors are sorted
    sorted_indices = np.argsort(A_evals)
    A_evecs = A_evecs[:,sorted_indices]

    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.5)
    x = np.arange(A_evals.size)

    # Smallest laplacian eigenvectors
    axs[0,0].scatter(x, L_evecs[:,0], s=0.1, c='red', label='Smallest')
    axs[0,0].scatter(x, L_evecs[:,1], s=0.1, c='blue', label='Second Smallest')
    axs[0,0].set_xlabel('Indices')
    axs[0,0].set_ylabel('Eigenvectors')
    axs[0,0].set_title(f'Smallest Laplacian Eigenvectors for {graph}')
    axs[0,0].legend()

    # Largest laplacian eigenvectors
    axs[0,1].scatter(x, L_evecs[:,-1], s=0.1, c='red', label='Largest')
    axs[0,1].scatter(x, L_evecs[:,-2], s=0.1, c='blue', label='Second Largest')
    axs[0,1].set_xlabel('Indices')
    axs[0,1].set_ylabel('Eigenvectors')
    axs[0,1].set_title(f'Largest Laplacian Eigenvectors for {graph}')
    axs[0,1].legend()

    # Smallest adjacency eigenvectors
    axs[1,0].scatter(x, A_evecs[:,0], s=0.1, c='red', label='Smallest')
    axs[1,0].scatter(x, A_evecs[:,1], s=0.1, c='blue', label='Second Smallest')
    axs[1,0].set_xlabel('Indices')
    axs[1,0].set_ylabel('Eigenvectors')
    axs[1,0].set_title(f'Smallest Adjacency Eigenvectors for {graph}')
    axs[1,0].legend()

    # Largest adjecency eigenvectors
    axs[1,1].scatter(x, A_evecs[:,-1], s=0.1, c='red', label='Largest')
    axs[1,1].scatter(x, A_evecs[:,-2], s=0.1, c='blue', label='Second Largest')
    axs[1,1].set_xlabel('Indices')
    axs[1,1].set_ylabel('Eigenvectors')
    axs[1,1].set_title(f'Largest Adjacency Eigenvectors for {graph} Graph')
    axs[1,1].legend()

    #plt.tight_layout()
    plt.show()
    # plt.savefig(f'images/{graph}.png')

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
    # b()

    L, A = spoke_and_wheel(150)
    L_evals, L_evecs = np.linalg.eig(L)
    # Ensure eigenvalues/eigenvectors are sorted
    sorted_indices = np.argsort(L_evals)
    L_evecs = L_evecs[:,sorted_indices]

    A_evals, A_evecs = np.linalg.eig(A)
    # Ensure eigenvalues/eigenvectors are sorted
    sorted_indices = np.argsort(A_evals)
    A_evecs = A_evecs[:,sorted_indices]

    fig, ax = plt.subplots(1)
    # Smallest laplacian eigenvectors
    ax.scatter(L_evecs[:,0], L_evecs[:,1], s=0.1)
    plt.show()


if __name__ == '__main__':
    main()