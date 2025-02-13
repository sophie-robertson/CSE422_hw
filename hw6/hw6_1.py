import numpy as np

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

def main():
    # test_graphs()

if __name__ == '__main__':
    main()