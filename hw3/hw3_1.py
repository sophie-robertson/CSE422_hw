import numpy as np
import matplotlib.pyplot as plt

# n = number of data points, d = dimensions
def training_data(theta_star, n=2000, d=20):
    X = np.random.normal(0,1, size=(n,d))
    y = X.dot(theta_star) + np.random.normal(0,0.5,size=(n,1))
    return X, y

# m = number of data points, d = dimensions
def test_data(m=2000, d=20):
    X = np.random.normal(0,1, size=(m,d))
    return X

def train_loss(X, y, theta): # change to mean to discuss for parts a-c
    return np.mean((X.dot(theta) - y) ** 2) # change to sum for full squared error

def test_loss(X, theta_star, theta_hat):
    return np.mean(((X.dot(theta_star)) - (X.dot(theta_hat))) ** 2)

def closed_form(X, y):
    theta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    return theta_hat, train_loss(X, y, theta_hat)

def gradient(X, y, theta): # remove division by n for full squared error
    n = X.shape[0]
    return (2 / n) * X.T @ (X @ theta - y)

def grad_descent(X_train, y_train, d=20, n=20):
    alpha_vals = [0.00007, 0.00035, 0.0007]
    losses = {}
    for step in alpha_vals:
        losses[step] = []
        theta = np.zeros((d, 1))
        for i in range(n):
            theta -= step * gradient(X_train, y_train, theta)
            losses[step].append(train_loss(X_train, y_train, theta))
    return losses

def plot_grad_descent(losses, n=20):
    fig, ax = plt.subplots()
    epochs = [i + 1 for i in range(n)]
    for step in losses.keys():
        ax.plot(epochs, losses[step], label=f'Step size {step}')
        # ax.fill_between(x, y1, y2, color='lightblue', alpha=0.5)
    ax.legend()
    ax.set_xlabel('Number of iterations')
    ax.set_xticks(epochs)
    ax.set_ylabel('Training loss')
    ax.set_title('Gradient descent for different step sizes')
    plt.show()


def main():
    # # Part (a)
    # print("Part (a)")
    d = 20
    theta_star = np.random.normal(0,1, size=(d,1)) # fix some true distribution
    # X_train, y_train= training_data(theta_star)
    # theta_hat, min_loss = closed_form(X_train, y_train)
    # print(f"Minimum loss: {min_loss}")
    # all_zeros_loss = train_loss(X_train, y_train, np.zeros((X_train.shape[1], 1)))
    # print(f"Loss with theta = 0: {all_zeros_loss}")
    # print()

    # # Part (b)
    # print("Part (b)")
    # X_test = test_data()
    # l_test = test_loss(X_test, theta_star, theta_hat)
    # print(f"Average test loss for m = 2000: {l_test}")
    # print()

    # m_range = [10**i for i in range(1, 7)]
    # for m in m_range:
    #     X_test = test_data(m)
    #     l_test = test_loss(X_test, theta_star, theta_hat)
    #     print(f"Average test loss for m = {m}: {l_test}")
    # print()

    # # Part (c)
    # print("Part (c)")
    # X_test = test_data()
    # for n in range(500, 2500, 500):
    #     X_train, y_train = training_data(theta_star, n=n)
    #     theta_hat, l_train = closed_form(X_train, y_train)
    #     l_test = test_loss(X_test, theta_star, theta_hat)
    #     param_dist = np.linalg.norm(theta_star - theta_hat)
    #     print(f"Parameter distance for n = {n}: {param_dist}")
    #     print(f"Training loss: {l_train}")
    #     print(f"Test loss: {l_test}")
    #     print(f"Parameter distance / Training loss: {param_dist / l_train}")
    #     print(f"Parameter distance / Test loss: {param_dist / l_test}")
    #     print()

    # Part (d)
    print("Part (d)") # hard to tell whats happening with total training loss- use average train loss instead?
    X_train, y_train= training_data(theta_star)
    losses = grad_descent(X_train, y_train)
    # print(losses)
    plot_grad_descent(losses)

if __name__ == '__main__':
    main()
