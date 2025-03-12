import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_parks():
    park_df = pd.read_csv("./data/parks.csv")
    return park_df

def distance(i, j, df):
    row_i = df.iloc[i]
    row_j = df.iloc[j]
    distance = np.sqrt((row_i["Longitude"] - row_j["Longitude"])**2 + (row_i["Latitude"] - row_j["Latitude"])**2)
    return distance

def tour_dist(df, permutation):
    total_dist = 0
    for k in range(len(permutation) - 1):
        i = permutation[k]
        j = permutation[k+1]
        total_dist += distance(i, j, df)
    # Dist from last park to first 
    total_dist += distance(permutation[-1], permutation[0], df)
    return total_dist

def mcmc(NUM_ITER, T, consecutive = True):
    df = load_parks()
    N = 30

    best_distance = float('inf')
    best_tour = None

    distances = []

    tour = np.random.permutation(N)
    # distances.append(tour_dist(df, tour))
    for i in range(NUM_ITER):
        new_tour = tour.copy()
        first = np.random.randint(0, N)
        if consecutive:
            second = first + 1 if first != N-1 else 0
        else:
            second = first
            while(second == first):
                second = np.random.randint(0, N)

        temp = tour[first]
        new_tour[first] = tour[second]
        new_tour[second] = temp

        curr_dist = tour_dist(df, tour)
        new_dist = tour_dist(df, new_tour)
        diff = new_dist - curr_dist
        
        if new_dist < best_distance:
            best_distance = new_dist
            best_tour = new_tour

        p = np.exp(-diff/T) if T > 0 else 0
        if diff < 0 or np.random.binomial(1, p):
            tour = new_tour
            distances.append(new_dist)
        else:
            distances.append(curr_dist)
            

    return tour, best_distance, best_tour, distances

def b():
    NUM_ITER = 12000
    Ts = [0, 1, 10, 100]
    fig, axs = plt.subplots(2, 2)
    for i, T in enumerate(Ts):
        print(T)
        row = int(i / 2)
        col = 0 if i % 2 == 0 else 1
        ax = axs[row, col]
        ax.set_title(f"Tour Distance vs. Iteration, T = {T}")
        for j in range(10):
            print("\t" + str(j))
            ending_point, best_dist, best_tour, dist = mcmc(NUM_ITER, T)
            ax.plot(range(NUM_ITER), dist)

    plt.show()
            

def c():
    NUM_ITER = 12000
    Ts = [0, 1, 10, 100]
    fig, axs = plt.subplots(2, 2)
    for i, T in enumerate(Ts):
        print(T)
        row = int(i / 2)
        col = 0 if i % 2 == 0 else 1
        ax = axs[row, col]
        ax.set_title(f"Tour Distance vs. Iteration, T = {T}")
        for i in range(10):
            print("\t" + str(i))
            ending_point, best_dist, best_tour, dist = mcmc(NUM_ITER, T, consecutive=False)
            ax.plot(range(NUM_ITER), dist)

    plt.show()


def main():
    #b()
    c()

if __name__ == '__main__':
    main()

