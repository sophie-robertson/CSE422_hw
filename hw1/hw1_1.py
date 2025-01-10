import numpy as np
import matplotlib.pyplot as plt

# Select one of the n bins uniformly at random
def uniform(m, n):
    bins = np.zeros(n)
    assignments = np.random.randint(low=0, high=n, size=(m,), dtype=int)
    for b in range(n):
        bins[b] = np.count_nonzero(assignments == b)
    return np.max(bins)

# Select two bins uniformly at random, and place the ball into whichever bin has fewer balls so far
# (breaking ties arbitrarily).
def two_choices(m, n):
    bins = np.zeros(n)
    for ball in range(m):
        two_bins = np.random.randint(low=0, high=n, size=(2,), dtype=int)
        bin_ind = two_bins[0] if bins[two_bins[0]] <= bins[two_bins[1]] else two_bins[1]
        bins[bin_ind] += 1
    return np.max(bins)

# Select three bins uniformly at random, and place the ball into whichever bin has fewer balls so far
# (breaking ties arbitrarily).
def three_choices(m, n):
    bins = np.zeros(n)
    for ball in range(m):
        three_bins = np.random.randint(low=0, high=n, size=(3,), dtype=int)
        counts = np.array([bins[three_bins[0]], bins[three_bins[1]], bins[three_bins[2]]])
        bin_ind = three_bins[np.argmin(counts)]
        bins[bin_ind] += 1
    return np.max(bins)

# Select one of the n bins Bi uniformly at random, and place the ball into either Bi or B(i+1) mod 
# n, whichever one has fewer balls so far (breaking ties arbitrarily).
def strategy_four(m, n):
    bins = np.zeros(n)
    assignments = np.random.randint(low=0, high=n, size=(m,), dtype=int)
    for a in assignments:
        bin_ind = a if bins[a] <= bins[(a + 1) % n] else (a + 1) % n
        bins[bin_ind] += 1
    return np.max(bins)


n_trials = 100
m = 1000000
n = 100000
uniform_maxload = np.zeros(n_trials)
twochoices_maxload = np.zeros(n_trials)
threechoices_maxload = np.zeros(n_trials)
stratfour_maxload = np.zeros(n_trials)
for j in range(n_trials):
    if j % 10 == 0:
        print("On iteration: " + str(j))
    uniform_maxload[j] = uniform(m, n)
    twochoices_maxload[j] = two_choices(m, n)
    threechoices_maxload[j] = three_choices(m, n)
    stratfour_maxload[j] = strategy_four(m, n)

fig, axs = plt.subplots(2, 2)
axs[0,0].hist(uniform_maxload)
axs[0,0].set_title("Uniform Sampling Max Load")
axs[0,0].set_xlabel("Value")
axs[0,0].set_ylabel("Frequency")

axs[0,1].hist(twochoices_maxload)
axs[0,1].set_title("Two Choices Sampling Max Load")
axs[0,1].set_xlabel("Value")
axs[0,1].set_ylabel("Frequency")

axs[1,0].hist(threechoices_maxload)
axs[1,0].set_title("Three Choices Sampling Max Load")
axs[1,0].set_xlabel("Value")
axs[1,0].set_ylabel("Frequency")

axs[1,1].hist(stratfour_maxload)
axs[1,1].set_title("Adjacent Choices Sampling Max Load")
axs[1,1].set_xlabel("Value")
axs[1,1].set_ylabel("Frequency")
plt.show()