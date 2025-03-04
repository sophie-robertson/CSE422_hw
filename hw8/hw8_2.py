import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_close(test_part_c = False):
    
    close_df = pd.read_csv("data/close.csv", header = None)
    ticker_names = []
    with open("data/tickers.csv") as file:
        for line in file:
            ticker_names.append(line.strip())
    close_df.columns = ticker_names
    if test_part_c:
        close_df["NVDA"] = 0
    return close_df

def a(test_part_c = False):
    eps = [0, 0.01, 0.1, 1, 2, 4, 8]
    df = load_close()
    # print(df)

    if test_part_c:
        percent_returns = df.pct_change()#.dropna()
        percent_returns["NVDA"] = 0
        percent_returns = percent_returns.fillna(0)
    else:
        percent_returns = df.pct_change().fillna(0)

    print(percent_returns)
    print(percent_returns.shape)

    #losses = -percent_returns
    N = percent_returns.shape[1]
    print(N)
    T = percent_returns.shape[0]
    print(T)
    eps_to_return = {}
    for e in eps:
        potential_vec = np.ones(N) / N
        cumulative_returns = [0]
        for t in range(1,T):
            returns = percent_returns.iloc[t, :]
            weighted_return = np.dot(potential_vec, returns)
            #print(weighted_return)
            cumulative_returns.append(cumulative_returns[-1] + (weighted_return))
            
            losses_t = -returns
            potential_vec *= np.exp(-e * losses_t)
            # Should this be normalized every time ? It should be a probability distribution right
            potential_vec /= potential_vec.sum()
    
        eps_to_return[e] = cumulative_returns

    fig, ax = plt.subplots()
    for e in eps_to_return.keys():
        print(e)
        ax.plot(range(T), eps_to_return[e], label = e)
    ax.set_xlabel("Days")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"Cumulative Return Using Multiplicative Weights with Different Epsilons\nNVIDIA Removed = {test_part_c}")
    ax.legend()
    plt.show()

def b(test_part_c = False):
    eps = [8] #[0, 0.01, 0.1, 1, 2, 4, 8]
    df = load_close(test_part_c)
    if test_part_c:
        percent_returns = df.pct_change()#.dropna()
        percent_returns["NVDA"] = 0
        percent_returns = percent_returns.dropna()
    else:
        percent_returns = df.pct_change().dropna()
    N = df.shape[1]
    T = df.shape[0]
    eps_to_max_prob = {}
    for e in eps:
        potential_vec = np.ones(N) / N
        max_probs = []
        for t in range(T-1):
            returns = percent_returns.iloc[t, :]
            max_probs.append(potential_vec[np.argmax(potential_vec)])
            # For seeing what the alg does
            print(f"Day: {t}, Stock: {df.iloc[:, np.argmax(potential_vec)].name}, Prob: {potential_vec[np.argmax(potential_vec)]}")

            losses_t = -returns
            potential_vec *= np.exp(-e * losses_t)
            potential_vec /= potential_vec.sum()
    
        eps_to_max_prob[e] = max_probs

    fig, axs = plt.subplots(4, 2)
    for i, e in enumerate(eps_to_max_prob.keys()):
        if i % 2 == 0:
            j = 0
        else:
            j = 1
        axs[int(i/2), j].plot(range(T-1), eps_to_max_prob[e], label = e)
        axs[int(i/2), j].set_title(f"Epsilon = {e}")

    fig.suptitle(f"Portfolio Diversity using Multiplicative Weights with Different Epsilons\nNVIDIA Removed = {test_part_c}")
    fig.supylabel("Maximum Probability Assigned")
    fig.supxlabel("Days")
    plt.subplots_adjust(hspace=0.8)
    plt.show()

    # Notes: All epsilons above 2 end up placing the maximum probability on NVIDIA (correctly)
        # As epsilon increases, the algorithm invests in NVIDIA earlier and with more 'certainty'
        # ex: at eps 8, the ending probability for NVIDIA is .999988... almost 100%


def main():
    #a()
    b()

    # For prblm 2 part c
    #a(test_part_c = True)
    #b(test_part_c = True)
    

if __name__ == '__main__':
    main()