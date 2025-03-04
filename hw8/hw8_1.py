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

def a_sam():
    df = load_close()
    # Get best-performing stock
    total_change = (df.iloc[-1] - df.iloc[0]) / df.iloc[0]
    best_performing = df.columns[np.argmax(total_change)]

    # start_date = '05/28/2019'
    # end_date = '05/24/2024'
    # date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    days = np.arange(df.shape[0])
    pct_change = df.pct_change().fillna(0) # 0% return on the first day
    pct_change = pct_change + 1 # Add 1 to express as a multiple of the original price
    # Daily returns for best-performing stock
    best_returns = pct_change[best_performing]
    cum_best = best_returns.cumprod() - 1 # Subtract 1 so that no change is a return of 0
    # Daily returns for equal investment
    equal_returns = pct_change.mean(axis=1)
    cum_equal = equal_returns.cumprod() - 1

    plt.plot(days, cum_best, label='Best Stock (NVDA)')
    plt.plot(days, cum_equal, label='Equal Investment Portfolio')
    plt.title('Cumulative returns of portfolios')
    plt.legend()
    plt.show()

def a_additive_return():
    df = load_close()
    # Get best-performing stock
    total_change = (df.iloc[-1] - df.iloc[0]) / df.iloc[0]
    best_performing = df.columns[np.argmax(total_change)]

    # start_date = '05/28/2019'
    # end_date = '05/24/2024'
    # date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    days = np.arange(df.shape[0])
    pct_change = df.pct_change().fillna(0) # 0% return on the first day
    # Daily returns for best-performing stock
    best_returns = pct_change[best_performing]
    cum_best = best_returns.cumsum() # Subtract 1 so that no change is a return of 0
    # Daily returns for equal investment
    equal_returns = pct_change.mean(axis=1)
    cum_equal = equal_returns.cumsum()

    plt.plot(days, cum_best, label='Best Stock (NVDA)')
    plt.plot(days, cum_equal, label='Equal Investment Portfolio')
    plt.title('Cumulative returns of portfolios')
    plt.legend()
    plt.show()

    
def c():
    # Is this asking for the inequality we discussed ? 
    # We make at most 2*sqrt(Tlog(N)) more mistakes than the best expert (best stock) where
        # T = number of days
        # N = number of experts (stocks)
    df = load_close()
    num_stocks = df.shape[1]
    num_days = df.shape[0]
    # Calculate rho
    # max_loss = np.zeros(num_stocks)
    # for j in range(num_days - 1):
    #     diff = (df.iloc[j] - df.iloc[j+1])/df.iloc[j]
    #     abs_diff = diff.abs()
    #     for i in range(num_stocks):
    #         if abs_diff[i] > max_loss[i]:
    #             max_loss[i] = abs_diff[i]
    # rho = np.max(max_loss).item()
    # print(f'rho = {rho}')

    # Alternate calculation
    changes = df.pct_change().dropna().abs()
    rho = df.max(changes) # Find maximum percent change
    print(f'rho = {rho}')


    epsilon = np.sqrt(np.log(num_stocks)/ num_days)
    print(f"Theoretical best epsilon: {epsilon}")

    regret = 2 * rho * np.sqrt(num_days * np.log(num_stocks))
    print(f'Regret bound: {regret}')

    # bound = 2 * np.sqrt(df.shape[0] * np.log(df.shape[1]))
    # print(f"Theoretical bound (We make at most this many more mistakes than the best stock): {bound}")

    # number of mistakes weighted majority ≤ 2.41(#mistakes expert i + log_2(N))
    # To achieve the lowest bound, we use expert i as the best performing stock 


    # To optimize epsilon, we calculate epsilon = sqrt(log(N)/T)
    # epsilon = np.sqrt(np.log(df.shape[1])/ df.shape[0])
    # print(f"Theoretical best epsilon: {epsilon}")

def d():
    # No 
    df = load_close()
    new_mat = np.random.choice([1, 0], size=(df.shape[0], df.shape[1]))
    new_mat[0, :] = np.ones(df.shape[1])
    total_returns = np.sum(new_mat, axis=0)
    print(f"Best performing stock: {df.index[np.argmax(total_returns)]}, with returns: {total_returns[np.argmax(total_returns)]}")
    # No real strategy as we can't rely on what has happened previously as any indicator for what will happen next. 
    return 

def d_sam():
    df = load_close()
    new_mat = np.random.choice([1, 0], size=(df.shape[0] - 1, df.shape[1])) # Number of changes for stocks
    # Get best-performing stock
    total_change = np.sum(new_mat, axis=0) # Best performing has the most 1s (doubling of stock value)
    best_performing = np.argmax(total_change)

    # Get cumulative returns (log_2 based, otherwise we get numerical overflow)
    days = np.arange(df.shape[0])
    # Daily returns for best-performing stock
    cum_best = new_mat[:,best_performing].cumsum()
    cum_best = np.insert(cum_best, 0, 0) # Add 0 to start

    plt.plot(days, cum_best)
    plt.title('Cumulative returns of best stock')
    plt.xlabel('Days')
    plt.ylabel('Returns')
    plt.show()

def main():
    # a()
    # a_sam()
    # a_additive_return()
    # c()
    # d()
    d_sam()
    

if __name__ == '__main__':
    main()