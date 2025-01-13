import numpy as np
import matplotlib.pyplot as plt

# Select one of the n bins uniformly at random
def uniform(m, n):
    bins = np.zeros(n)
    for ball in range(m):
        bin1 = np.random.randint(0, n)
        bins[bin1] += 1
    return np.max(bins)

# Select two bins uniformly at random (implemented without replacement), and place the ball into whichever bin has fewer balls so far
# (breaking ties arbitrarily).
def two_choices(m, n):
    bins = np.zeros(n)
    for ball in range(m):
        bin1 = np.random.randint(0, n)
        bin2 = bin1
        while bin2 == bin1:
            bin2 = np.random.randint(0,n)
        bin_ind = bin1 if bins[bin1] <= bins[bin2] else bin2
        bins[bin_ind] += 1
    return np.max(bins)

# Select three bins uniformly at random (implemented without replacement), and place the ball into whichever bin has fewer balls so far
# (breaking ties arbitrarily).
def three_choices(m, n):
    bins = np.zeros(n)
    sampler = np.arange(n)
    for ball in range(m):
        bin1 = np.random.randint(0, n)
        bin2 = bin1
        bin3 = bin1
        while bin2 == bin1:
            bin2 = np.random.randint(0,n)
        while bin3 == bin1 or bin3 == bin2:
            bin3 = np.random.randint(0,n)
        three_bins = [bin1, bin2, bin3]
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
    print("On iteration: " + str(j))
    uniform_maxload[j] = uniform(m, n)
    print("\tUniform done")
    twochoices_maxload[j] = two_choices(m, n)
    print("\tTwo choices done")
    threechoices_maxload[j] = three_choices(m, n)
    print("\tThree choices done")
    stratfour_maxload[j] = strategy_four(m, n)
    print("\tAdjacent done")

print(uniform_maxload)
print(twochoices_maxload)
print(threechoices_maxload)
print(stratfour_maxload)

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