import numpy as np
import matplotlib.pyplot as plt

# creates transition matrix P for an n-cycle
def cycle(n):
    P = np.zeros((n,n))
    for i in range(n):
        P[i][(i+1) % n] = 1/2
        P[i][(i-1) % n] = 1/2
    stat = np.full(n, 1 / n)
    return P, stat

# creates transition matrix P for an n-cycle
# with self-loops on even vertices
def cycle_with_loops(n):
    P = np.zeros((n,n))
    stat = np.zeros(n)
    for i in range(n):
        if i % 2 == 0: # odd vertex
            P[i][(i+1) % n] = 1/2
            P[i][(i-1) % n] = 1/2
            stat[i] = 4/(5*n)
        else: # even vertex
            P[i][(i+1) % n] = 1/3
            P[i][i] = 1/3
            P[i][(i-1) % n] = 1/3
            stat[i] = 6/(5*n)
    return P, stat

def test_graphs():
    P, stat = cycle(6)
    print(P)
    print(stat)
    P, stat = cycle_with_loops(6)
    print(P)
    print(stat)

# p1, p2 numpy vectors of the same length
def TV(p1, p2):
    return (1/2) * np.sum(np.abs(p1 - p2))

def b():
    P_1, stat_1 = cycle(17)
    var_1 = TV_walk(17, P_1, stat_1, 0)
    P_2, stat_2 = cycle(18)
    var_2 = TV_walk(18, P_2, stat_2, 0)
    P_3, stat_3 = cycle_with_loops(18)
    var_3 = TV_walk(18, P_3, stat_3, 0)

    steps = np.arange(100 + 1)
    plt.plot(steps, var_1, label='17-cycle')
    plt.plot(steps, var_2, label='18-cycle')
    plt.plot(steps, var_3, label='18-cycle with loops')
    plt.legend()
    plt.savefig('figures/Q1b.png')
    plt.show()

def TV_walk(n, P, stat, start):
    curr = np.zeros(n)
    curr[start] = 1
    variations = []
    for t in range(100+1):
        variations.append(TV(curr, stat).item())
        curr = curr @ P
    return variations

def test_TV_walk():
    P, stat = e(5)
    print(P)
    # variations = TV_walk(18, P, stat)
    # print(variations)

def c():
    P_1, _ = cycle(17)
    print(P_1)
    calculate_eigvals(P_1)
    print()
    P_2, _ = cycle(18)
    calculate_eigvals(P_2)
    print()
    P_3, _ = cycle_with_loops(18)
    calculate_eigvals(P_3)

def calculate_eigvals(P):
    P_evals, P_evecs = np.linalg.eig(np.transpose(P))
    # sorted_indices = np.argsort(P_evals)
    # P_evecs = P_evecs[:, sorted_indices]
    # P_evals = P_evals[sorted_indices]
    P_evals = np.sort(P_evals)
    print(f'Smallest eigenvalue: {P_evals[0]}')
    print(f'Second largest eigenvalue: {P_evals[-2]}')
    # print(f'Largest eigenvalue: {P_evals[-1]}')
    # print()
    # print(P_evals[-1])
    # print(P_evecs[-1]) # not giving the correct stationary distributions - should be uniform for 1/3

def e():
    n = 18
    # Cycles for vertices i % 3 != 0, cross-graph connections
    # for vertices with i % 3 == 0.
    P = np.zeros((n,n))
    for i in range(n): # Cycles
        P[i][(i+1) % n] = 1/3
        P[i][(i-1) % n] = 1/3
        P[i][i] = 1/3
    for i in range(0, n//2, 3):
        P[i][i + n//2] = 1/3
        P[i + n//2][i] = 1/3
        P[i][i] = 0
        P[i+n//2][i+n//2] = 0
    print(P)

    # # Self loops on every vertex
    # P = np.zeros((n,n))
    # for i in range(n): # Cycles
    #     P[i][(i+1) % n] = 1/3
    #     P[i][(i-1) % n] = 1/3
    #     P[i][i] = 1/3

    # # Self loops - performs identically to the 18 w loops, 0.9549814862695705
    # P = np.zeros((n,n))
    # for i in range(n):
    #     P[i][(i+1) % n] = 1/3
    #     P[i][(i-1) % n] = 1/3
    #     P[i][i] = 1/3

    stat = np.full(n, 1 / n)
    calculate_eigvals(P)

    for i in range(3):
        var = TV_walk(18, P, stat, i)
        
        P_1, stat_1 = cycle(17)
        var_1 = TV_walk(17, P_1, stat_1, i)
        P_2, stat_2 = cycle(18)
        var_2 = TV_walk(18, P_2, stat_2, i)
        P_3, stat_3 = cycle_with_loops(18)
        var_3 = TV_walk(18, P_3, stat_3, i)

        steps = np.arange(100 + 1)
        plt.title(f'TV Distance for starting vertex v = {i + 1}')
        plt.plot(steps, var, label='Constructed')
        plt.plot(steps, var_1, label='17-cycle')
        plt.plot(steps, var_2, label='18-cycle')
        plt.plot(steps, var_3, label='18-cycle with loops')
        plt.legend()
        plt.savefig(f'figures/Q1e_{i}.png')
        plt.show()

def main():
    # test_graphs()
    # test_TV_walk()
    # b()
    # c()
    e()

if __name__ == '__main__':
    main()