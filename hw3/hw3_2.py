import numpy as np
import matplotlib.pyplot as plt


def training_data(train_n = 200, test_n = 2000, d = 200):
    X_train = np.random.normal(0,1, size=(train_n,d))
    theta_star = np.random.normal(0,1, size=(d,1))
    y_train = X_train.dot(theta_star) + np.random.normal(0,0.5,size=(train_n,1))
    X_test = np.random.normal(0,1, size=(test_n,d))
    y_test = X_test.dot(theta_star) + np.random.normal(0,0.5,size=(test_n,1))
    return X_train, y_train, X_test, y_test

def train_loss(X, y, theta): # change this for part 2
    return np.mean((X.dot(theta) - y) ** 2) # change to sum for full squared error

def test_loss(X, theta_star, theta_hat):
    return np.mean(((X.dot(theta_star)) - (X.dot(theta_hat))) ** 2)

def closed_form(X, y):
    theta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    return theta_hat, train_loss(X, y, theta_hat)

def closed_form_ridge(X, y):
    lambdas = [0.0005, 0.005, 0.05, 0.5, 5, 50, 50]
    m = X.shape[0]
    regularized = {}
    for l in lambdas:
        regularized[l] = np.linalg.solve(X.T @ X + l * np.eye(m), X.T @ y)
    train_losses = np.zeros(len(regularized))
    for i, theta_hat in enumerate(regularized.values()):
        train_losses[i] = (train_loss(X, y, theta_hat))

    return regularized, train_losses

def main():
    # # Part (a)
    # print("Part (a)")
    # d = 20
    # X_train, y_train, X_test, y_test= training_data()
    # theta_hat, min_loss = closed_form(X_train, y_train)
    # print(f"Minimum loss: {min_loss}")
    # all_zeros_loss = train_loss(X_train, y_train, np.zeros((X_train.shape[1], 1)))
    # print(f"Loss with theta = 0: {all_zeros_loss}")
    # print()

    # Part (b)
    print("Part (b)")
    d = 20
    X_train, y_train, X_test, y_test= training_data()
    sols, losses = closed_form_ridge(X_train, y_train)
    print(f"Minimum loss: {losses}")
    all_zeros_loss = train_loss(X_train, y_train, np.zeros((X_train.shape[1], 1)))
    print(f"Loss with theta = 0: {all_zeros_loss}")
    print()


if __name__ == '__main__':
    main()