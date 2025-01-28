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
    return np.sum((X.dot(theta) - y) ** 2)
    return np.mean((X.dot(theta) - y) ** 2) # if average squared error

def test_loss(X, theta_star, theta_hat):
    return np.mean(((X.dot(theta_star)) - (X.dot(theta_hat))) ** 2)

def closed_form(X, y):
    theta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    return theta_hat, train_loss(X, y, theta_hat)

def gradient(X, y, theta): # remove division by n for full squared error
    n = X.shape[0]
    return 2 * X.T @ (X @ theta - y)
    return (2 / n) * X.T @ (X @ theta - y) # iff average squared error

def grad_descent(X_train, y_train, d=20, t=20):
    alpha_vals = [0.00007, 0.00035, 0.0007]
    losses = {}
    for step in alpha_vals:
        losses[step] = []
        theta = np.zeros((d, 1))
        losses[step].append(train_loss(X_train, y_train, theta))
        for i in range(t):
            theta -= step * gradient(X_train, y_train, theta)
            losses[step].append(train_loss(X_train, y_train, theta))
    return losses

def SGD(X_train, y_train, d=20, t=20000):
    alpha_vals = [0.0005, 0.005, 0.01]
    losses = {}
    for step in alpha_vals:
        losses[step] = []
        theta = np.zeros((d, 1))
        losses[step].append(train_loss(X_train, y_train, theta))
        for j in range(t):
            i = np.random.randint(X_train.shape[0])
            grad = (X_train[i,:].dot(theta) - y_train[i]) * X_train[i,:]
            theta -= step * np.expand_dims(grad, 1)
            losses[step].append(train_loss(X_train, y_train, theta))
    return losses

def SGD_var(X_train, y_train, min_loss, d=20, t=20000):
    alpha = 0.01
    losses = []
    theta = np.zeros((d, 1))
    losses.append(train_loss(X_train, y_train, theta))
    conv_rate = -1
    for j in range(t):
        i = np.random.randint(X_train.shape[0])
        grad = (X_train[i,:].dot(theta) - y_train[i]) * X_train[i,:]
        theta -= alpha * np.expand_dims(grad, 1)
        alpha = alpha * (999/1000)
        losses.append(train_loss(X_train, y_train, theta))
        if losses[-1] < min_loss and conv_rate < 0:
            conv_rate = j
    return losses, conv_rate

def plot_grad_descent(losses, t, type='Gradient descent'):
    fig, ax = plt.subplots()
    epochs = [i for i in range(t + 1)]
    for step in losses.keys():
        ax.plot(epochs, losses[step], label=f'Step size {step}')
    ax.legend()
    ax.set_xlabel('Number of iterations')
    if t == 20: ax.set_xticks(epochs)
    ax.set_ylabel('Training loss')
    ax.set_title(f'{type} for different step sizes')
    plt.show()


def main():
    # Part (a)
    print("Part (a)")
    d = 20
    theta_star = np.random.normal(0,1, size=(d,1)) # fix some true distribution
    X_train, y_train= training_data(theta_star)
    theta_hat, min_loss = closed_form(X_train, y_train)
    print(f"Minimum loss: {min_loss}")
    all_zeros_loss = train_loss(X_train, y_train, np.zeros((X_train.shape[1], 1)))
    print(f"Loss with theta = 0: {all_zeros_loss}")
    print()
    X_train, y_train= training_data(theta_star)
    theta_hat, min_loss = closed_form(X_train, y_train)
    print(f"Minimum loss: {min_loss}")
    all_zeros_loss = train_loss(X_train, y_train, np.zeros((X_train.shape[1], 1)))
    print(f"Loss with theta = 0: {all_zeros_loss}")
    print()

    # Part (b)
    print("Part (b)")
    X_test = test_data()
    l_test = test_loss(X_test, theta_star, theta_hat)
    print(f"Average test loss for m = 2000: {l_test}")
    print()

    m_range = [10**i for i in range(1, 7)]
    for m in m_range:
        X_test = test_data(m)
        l_test = test_loss(X_test, theta_star, theta_hat)
        print(f"Average test loss for m = {m}: {l_test}")
    print()

    # Part (c)
    print("Part (c)")
    X_test = test_data()
    for n in range(500, 2500, 500):
        X_train, y_train = training_data(theta_star, n=n)
        theta_hat, l_train = closed_form(X_train, y_train)
        l_test = test_loss(X_test, theta_star, theta_hat)
        param_dist = np.linalg.norm(theta_star - theta_hat)
        print(f"Parameter distance for n = {n}: {param_dist}")
        print(f"Training loss: {l_train}")
        print(f"Test loss: {l_test}")
        print(f"Parameter distance / Training loss: {param_dist / l_train}")
        print(f"Parameter distance / Test loss: {param_dist / l_test}")
        print()

    # Part (d)
    print("Part (d)") # hard to tell whats happening with total training loss- use average train loss instead?
    X_train, y_train= training_data(theta_star)
    losses = grad_descent(X_train, y_train, t=20) # HUGE error for t=20
    for step in losses:
        print(f'Step size {step}, final loss {losses[step][-1]}')
    plot_grad_descent(losses, t=20)
    print()

    # Part (f)
    print("Part (f)")
    losses = SGD(X_train, y_train)
    for step in losses:
        print(f'Step size {step}, final loss {losses[step][-1]}')
    plot_grad_descent(losses, t=20000, type='SGD')
    min_loss = losses[0.0005][-1]
    print()

    # Part (g)
    print("Part (g)")
    var_loss, conv_rate = SGD_var(X_train, y_train, min_loss)
    comp_losses = {'0.01': losses[0.01], '0.0005': losses[0.0005], 'var': var_loss}
    for step in comp_losses:
        print(f'Step size {step}, final loss {comp_losses[step][-1]}')
    print(f"Convergence rate {conv_rate}")
    plot_grad_descent(comp_losses, t=20000, type='SGD')

if __name__ == '__main__':
    main()
