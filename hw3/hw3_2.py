import numpy as np
import matplotlib.pyplot as plt

def SGD(X_train, y_train, X_test, y_test, d=20, t=20000, alpha_vals = [0.0005, 0.005, 0.01], collect_thetas = False):
    tr_losses = {}
    te_losses = {}
    thetas = []
    theta_dict = {}
    for step in alpha_vals:
        tr_losses[step] = []
        te_losses[step] = []
        
        theta = np.zeros((d, 1))
        tr_losses[step].append(train_loss(X_train, y_train, theta))
        te_losses[step].append(test_loss(X_test, y_test, theta))
        if collect_thetas:
            theta_dict[step] = []
            theta_dict[step].append(np.linalg.norm(theta, ord = 2))
        for j in range(t):
            i = np.random.randint(X_train.shape[0])
            grad = (X_train[i,:].dot(theta) - y_train[i]) * X_train[i,:]
            theta -= step * np.expand_dims(grad, 1)
            tr_losses[step].append(train_loss(X_train, y_train, theta))
            te_losses[step].append(test_loss(X_test, y_test, theta))
            if collect_thetas:
                theta_dict[step].append(np.linalg.norm(theta, ord = 2))
        thetas.append(theta)
    return thetas, tr_losses, te_losses, theta_dict    

def training_data(train_n = 200, test_n = 2000, d = 200):
    X_train = np.random.normal(0,1, size=(train_n,d))
    theta_star = np.random.normal(0,1, size=(d,1))
    y_train = X_train.dot(theta_star) + np.random.normal(0,0.5,size=(train_n,1))
    X_test = np.random.normal(0,1, size=(test_n,d))
    y_test = X_test.dot(theta_star) + np.random.normal(0,0.5,size=(test_n,1))
    return X_train, y_train, X_test, y_test

def train_loss(X, y, theta): 
    # linalg norm returns 1 element array
    return np.linalg.norm((X.dot(theta) - y), ord=2, axis = 0) / np.linalg.norm(y, ord=2, axis = 0)

def test_loss(X_test, y_test, theta_hat):
    return np.linalg.norm((X_test.dot(theta_hat) - y_test), ord=2, axis = 0) / np.linalg.norm(y_test, ord=2, axis = 0)

def closed_form(X, y):
    theta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    return theta_hat, train_loss(X, y, theta_hat)

def closed_form_ridge(X, y):
    lambdas = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]
    m = X.shape[0]
    thetas = []
    for l in lambdas:
        thetas.append(np.linalg.solve(X.T @ X + l * np.eye(m), X.T @ y))
    train_losses = np.zeros(len(thetas))
    for i, theta_hat in enumerate(thetas):
        train_losses[i] = train_loss(X, y, theta_hat)[0]

    return thetas, train_losses



def a():
    print("Part (a)")
    n_trials = 10
    train_losses = np.zeros(n_trials)
    test_losses = np.zeros(n_trials)
    for i in range(n_trials):
        X_train, y_train, X_test, y_test= training_data()
        theta_hat, train_loss = closed_form(X_train, y_train)
        tl = test_loss(X_test, y_test, theta_hat)[0]
        train_losses[i] = train_loss[0]
        test_losses[i] = tl
    print(f"Average train loss: {np.mean(train_losses)}")
    print(f"Average test loss: {np.mean(test_losses)}")

def b():
    print("Part (b)")
    n_trials = 10
    lambdas = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]
    n_lambdas = len(lambdas)
    train_losses = np.zeros(n_lambdas)
    test_losses = np.zeros(n_lambdas)
    for i in range(n_trials):
        X_train, y_train, X_test, y_test= training_data()
        thetas, losses = closed_form_ridge(X_train, y_train)
        train_losses = np.add(train_losses, losses)
        for i, t in enumerate(thetas):
            # bc this is a 1 element array, take out the first
            test_losses[i] += test_loss(X_test, y_test, t)[0]
    averaged_train = train_losses/n_trials
    averaged_test = test_losses/n_trials
    plt.figure()
    plt.xscale("log")
    plt.xlabel("Lambda Value (Log Scale)")
    plt.ylabel("Normalized Loss")
    plt.plot(lambdas, averaged_train, label = "Train loss")
    plt.plot(lambdas, averaged_test, label = "Test loss")
    plt.legend()
    plt.show()

def c():
    n_trials = 10
    alphas = [0.00005, 0.0005, 0.005]
    temp_train_losses = np.zeros((len(alphas), n_trials))
    temp_test_losses = np.zeros((len(alphas), n_trials))
    steps = 1000000
    for i in range(n_trials):
        print(f"Trial {i}")
        X_train, y_train, X_test, y_test= training_data()
        thetas, _, _, _ = SGD(X_train, y_train, X_test, y_test, d=200, t=steps, alpha_vals=alphas)
        for a in range(len(alphas)):
            temp_train_losses[a, i] = np.add(temp_train_losses[a, i], train_loss(X_train, y_train, thetas[a]))
            temp_test_losses[a, i] = np.add(temp_test_losses[a, i], test_loss(X_test, y_test, thetas[a]))
        print(temp_train_losses[:, i])
        print(temp_test_losses[:, i])
    average_train = np.mean(temp_train_losses, axis = 1)
    average_test = np.mean(temp_test_losses, axis = 1)
    print(f"Average train losses: {average_train}")
    print(f"Average test losses: {average_test}")

def d():
    n_trials = 1
    alphas = [0.00005, 0.0005, 0.005]
    steps = 1000000
    theta_norms = {}
    for i in range(n_trials):
        print(f"Trial {i}")
        X_train, y_train, X_test, y_test= training_data()
        thetas, train_losses, test_losses, theta_norms = SGD(X_train, y_train, X_test, y_test, d=200, t=steps, alpha_vals=alphas, collect_thetas=True)

    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("Normalized Loss")
    for a in train_losses.keys():
        plt.plot(range(steps + 1), train_losses[a], label = f"Train loss, a = {a}")
        plt.plot(range(steps + 1), test_losses[a], label = f"Test loss, a = {a}")
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("L2 Norm of Theta")
    for a in train_losses.keys():
        plt.plot(range(steps + 1), theta_norms[a], label = f"Theta norms, a = {a}")
    plt.legend()
    plt.show()




def main():
    # # Part (a)
    # a()

    # # Part (b)
    #b()

    # Part (c)
    #c()

    d()



if __name__ == '__main__':
    main()