import numpy as np


# Get training data
d = 20 # dimensions of data
n = 2000 # number of data points

def training_data():
    X = np.random.normal(0,1, size=(n,d))
    theta_star = np.random.normal(0,1, size=(d,1))
    y = X.dot(theta_star) + np.random.normal(0,0.5,size=(n,1))
    return X, y

def loss(X, y, theta):
    return np.sum((X @ theta - y) ** 2)

def closed_form(X, y):
    theta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    return loss(X, y, theta_hat)
