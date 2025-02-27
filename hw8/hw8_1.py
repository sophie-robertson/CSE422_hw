# The ith column of close.csv should be a length 1259 vector representing the close 
# prices for the ith stock in tickers.csv for each day the stock market is open from 
# May 28th, 2019 to May 24th, 2024.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_close():
    close_df = pd.read_csv("data/close.csv", header = None)
    ticker_names = pd.read_csv("data/tickers.csv", header = None)
    close_df.columns = ticker_names
    return close_df

def a():
    df = load_close()
    daily_percent_changes = df.pct_change().dropna()
    percent_changes = daily_percent_changes + 1
    percent_changes.iloc[0] = np.ones(percent_changes.shape[1])
    total_returns = np.prod(percent_changes, axis=0)
    print(f"Best performing stock: {total_returns.index[np.argmax(total_returns)]}, with returns: {total_returns.iloc[np.argmax(total_returns)]}")

    equal_weight = daily_percent_changes + 1
    equal_weight.iloc[0] = np.ones(equal_weight.shape[1]) * (1/419)
    over_time = np.prod(equal_weight, axis=0)
    total = np.sum(over_time)
    print(f"Equal weight investment returns: {total}")
    return

def c():
    # Is this asking for the inequality we discussed ? 
    # We make at most 2*sqrt(Tlog(N)) more mistakes than the best expert (best stock) where
        # T = number of days
        # N = number of experts (stocks)
    df = load_close()
    bound = 2 * np.sqrt(df.shape[0] * np.log(df.shape[1]))
    print(f"Theoretical bound (We make at most this many more mistakes than the best stock): {bound}")

    # number of mistakes weighted majority ≤ 2.41(#mistakes expert i + log_2(N))
    # To achieve the lowest bound, we use expert i as the best performing stock 


    # To optimize epsilon, we calculate epsilon = sqrt(log(N)/T)
    epsilon = np.sqrt(np.log(df.shape[1])/ df.shape[0])
    print(f"Theoretical best epsilon: {epsilon}")

def d():
    # No 
    df = load_close()
    new_mat = np.random.choice([1, 0], size=(df.shape[0], df.shape[1]))
    new_mat[0, :] = np.ones(df.shape[1])
    total_returns = np.sum(new_mat, axis=0)
    print(f"Best performing stock: {df.index[np.argmax(total_returns)]}, with returns: {total_returns[np.argmax(total_returns)]}")
    # No real strategy as we can't rely on what has happened previously as any indicator for what will happen next. 
    return 


def main():
    # a()
    #   c isn't done but I'll come back to it 
    # c()
    d()
    

if __name__ == '__main__':
    main()