import numpy as np
import matplotlib.pyplot as plt

# Get data
train_n = 200
test_n = 2000
d = 300
X_train = np.random.normal(0,1, size=(train_n,d))
theta_star = np.random.normal(0,1, size=(d,1))
y_train = X_train.dot(theta_star) + np.random.normal(0,0.5,size=(train_n,1))
X_test = np.random.normal(0,1, size=(test_n,d))
y_test = X_test.dot(theta_star) + np.random.normal(0,0.5,size=(test_n,1))

def closed_form(X, y, _lambda):
    theta_hat = np.linalg.solve(X.T @ X + np.eye(X.shape[1]) * _lambda, X.T @ y)
    return theta_hat

def L1_gradient(X_i, y_i, theta, _lambda):
    sign = np.where(theta < 0, -1, np.where(theta > 0, 1, 0))
    return (X_i.T.dot(theta) - y_i) * X_i + sign

def L2_gradient(X_i, y_i, theta, _lambda): # for SGD
    return (X_i.T.dot(theta) - y_i) * X_i + 2 * _lambda * theta
    return (X.T @ X + 2 * _lambda * np.eye(X.shape[1])) @ theta - 2 * X.T @ y

# def test_loss(X_test, theta): # average mean squared
#     return np.mean(((X_test.dot(theta_star)) - (X_test.dot(theta))) ** 2)

def test_loss(X_test, y_test, theta): # normalized test loss
    return np.linalg.norm((X_test.dot(theta) - y_test), ord=2, axis = 0) / np.linalg.norm(y_test, ord=2, axis = 0)

def SGD_L1(X_train, y_train, t=500000, alpha_vals = [0.00005, 0.0005, 0.005, 0.05, 0.5], lambda_vals = [0.0005, 0.005, 0.05, 0.5, 5]):
    for reg in lambda_vals:
        for step in alpha_vals:
            theta = np.random.rand(X_train.shape[1], 1)
            for j in range(t):
                i = np.random.randint(X_train.shape[0])
                grad = L1_gradient(np.expand_dims(X_train[i,:],1), y_train[i], theta, 0)
                # print(grad.shape)
                theta -= step * grad
            loss = test_loss(X_test, y_test, theta).item()
            print(f"Step size: {step}, Lambda: {reg}, Test Loss: {loss}")
            # print(f"Step size: {step}, Test Loss: {loss}")


def SGD_L2(X_train, y_train, t=500000, alpha_vals = [0.00005, 0.0005, 0.005], lambda_vals = [0.0005, 0.005, 0.05, 0.5]):
    for reg in lambda_vals:
        theta = closed_form(X_train, y_train, reg)
        loss = test_loss(X_test, y_test, theta).item()
        print(f"Closed form loss for lambda = {reg}: {loss}")
        for step in alpha_vals:
            theta = closed_form(X_train, y_train, reg)
            for j in range(t):
                i = np.random.randint(X_train.shape[0])
                grad = train_loss(np.expand_dims(X_train[i,:],1), y_train[i], theta, 0)
                # print(grad.shape)
                theta -= step * grad
            loss = test_loss(X_test, y_test, theta).item()
            print(f"Step size: {step}, Lambda: {reg}, Test Loss: {loss}")
            # print(f"Step size: {step}, Test Loss: {loss}")

#SGD_L2(X_train, y_train)
SGD_L1(X_train, y_train)

