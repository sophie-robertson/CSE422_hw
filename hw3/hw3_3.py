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

# Gives closed form solution to ridge regression
def L2_closed_form(X, y, _lambda):
    theta_hat = np.linalg.solve(X.T @ X + np.eye(X.shape[1]) * _lambda, X.T @ y)
    return theta_hat

# X_i = row of X, y_i = value in y
# Gives gradient of loss function with L1 regularization
def L1_gradient(X_i, y_i, theta, _lambda):
    sign = np.where(theta < 0, -1, np.where(theta > 0, 1, 0)) # gradient of L1
    return (X_i.T.dot(theta) - y_i) * X_i + sign

# X_i = row of X, y_i = value in y
# Gives gradient of loss function with L2 regularization
def L2_gradient(X_i, y_i, theta, _lambda): # for SGD
    return (X_i.dot(theta) - y_i).dot(X_i) + 2 * _lambda * theta
    return (X.T @ X + 2 * _lambda * np.eye(X.shape[1])) @ theta - 2 * X.T @ y

def reg_gradient(X_i, y_i, theta):
    a = X_i.dot(theta) - y_i
    # print(a.shape)
    # print(X_i.shape)
    return a.T.dot(X_i)

# def test_loss(X_test, theta): # average mean squared
#     return np.mean(((X_test.dot(theta_star)) - (X_test.dot(theta))) ** 2)

def test_loss(X_test, y_test, theta): # normalized test loss
    return np.linalg.norm((X_test.dot(theta) - y_test), ord=2, axis = 0) / np.linalg.norm(y_test, ord=2, axis = 0)

# SGD using the L1 norm loss function
def SGD_L1(X_train, y_train, t=500000, alpha_vals = [0.00005, 0.0005, 0.005, 0.05, 0.5], lambda_vals = [0.0005, 0.005, 0.05, 0.5, 5]):
    for reg in lambda_vals:
        print(f"Lambda = {reg}")
        theta = L2_closed_form(X_train, y_train, reg)
        print(f"L2 closed form loss (baseline): {test_loss(X_test, y_test, theta).item()}")
        for step in alpha_vals:
            # theta = np.random.rand(X_train.shape[1], 1)
            theta = np.zeros((X_train.shape[1], 1))
            for j in range(t):
                i = np.random.randint(X_train.shape[0])
                grad = L1_gradient(np.expand_dims(X_train[i,:],1), y_train[i], theta, 0)
                # print(grad.shape)
                theta -= step * grad
            loss = test_loss(X_test, y_test, theta).item()
            print(f"Step size: {step}, Test Loss: {loss}")
            # print(f"Step size: {step}, Test Loss: {loss}")

def SGD_on_L2(X_train, y_train, t=500000, alpha_vals = [0.00005, 0.0005, 0.005], lambda_vals = [0.0005, 0.005, 0.05, 0.5]):
    for reg in lambda_vals:
        print(f"Lambda = {reg}")
        theta = L2_closed_form(X_train, y_train, reg)
        loss = test_loss(X_test, y_test, theta).item()
        print(f"Closed form loss for lambda = {reg}: {loss}")
        for step in alpha_vals:
            theta = L2_closed_form(X_train, y_train, reg) 
            for j in range(t):
                i = np.random.randint(X_train.shape[0])
                grad = reg_gradient(np.expand_dims(X_train[i,:],1), y_train[i], theta, 0)
                # print(grad.shape)
                theta -= step * grad
            loss = test_loss(X_test, y_test, theta).item()
            print(f"Step size: {step}, Test Loss: {loss}")

# Selects best lambda for closed form ridge regression via averaging over 1000 pulls from the distribution
def find_min_lambda(lambda_vals = [0.0005, 0.005, 0.05, 0.5]):
    min_loss = 1000
    min_lambda = None
    for reg in lambda_vals:
        loss = 0
        for i in range(1000):
            train_n = 200
            test_n = 2000
            d = 300
            X_train = np.random.normal(0,1, size=(train_n,d))
            theta_star = np.random.normal(0,1, size=(d,1))
            y_train = X_train.dot(theta_star) + np.random.normal(0,0.5,size=(train_n,1))
            X_test = np.random.normal(0,1, size=(test_n,d))
            y_test = X_test.dot(theta_star) + np.random.normal(0,0.5,size=(test_n,1))
            theta = L2_closed_form(X_train, y_train, reg)
            loss += test_loss(X_test, y_test, theta).item()
        loss /= 1000
        if loss < min_loss:
            min_loss = loss
            min_lambda = reg
    print(f'Min lambda: {min_lambda}, min_loss: {min_loss}')

# Tests L2_then_SGD model for given step sizes and decrement values over 10 pulls from the distribution
def L2_then_SGD(t=200000, alpha_vals = [0.0005, 0.005, 0.01, 0.05], lambda_vals = [0.5], dec_vals = [0.9, 0.99, 0.999]):
    train_n = 200
    test_n = 2000
    d = 300
    for reg in lambda_vals:
        # theta = L2_closed_form(X_train, y_train, reg)
        # closed_form_loss = test_loss(X_test, y_test, theta).item()
        # print(f"Closed form loss for lambda = {reg}: {loss}")
        for step in alpha_vals:
            init_step = step
            for dec in dec_vals:
                t_loss = 0
                closed_form_loss = 0
                for i in range(10): # check for 10 initializations
                    X_train = np.random.normal(0,1, size=(train_n,d))
                    theta_star = np.random.normal(0,1, size=(d,1))
                    y_train = X_train.dot(theta_star) + np.random.normal(0,0.5,size=(train_n,1))
                    X_test = np.random.normal(0,1, size=(test_n,d))
                    y_test = X_test.dot(theta_star) + np.random.normal(0,0.5,size=(test_n,1))

                    theta = L2_closed_form(X_train, y_train, reg) # start at closed form solution, then wander
                    closed_form_loss += test_loss(X_test, y_test, theta).item()

                    for j in range(t):
                        i = np.random.randint(X_train.shape[0], size=1)
                        grad = reg_gradient(X_train[i,:], y_train[i], theta).T
                        # print(grad.shape)
                        theta -= step * grad
                        step = step * dec
                    t_loss += test_loss(X_test, y_test, theta).item()

                closed_form_loss /= 10
                t_loss /= 10
                print(f"Initial step size: {init_step}, Decrement: {dec}, Test Loss: {t_loss}, Closed form: {closed_form_loss}")
                print('Better') if t_loss < closed_form_loss else print('Worse')
            print()


# find_min_lambda()
L2_then_SGD()
# SGD_on_L2(X_train, y_train)
# SGD_L1(X_train, y_train)

# Testing early stopping

X_val = X_train[:20, :]
y_val = y_train[:20, :]
X_train = X_train[20:, :]
y_train = y_train[20:, :]

# with variable step size
def early_stopping(t=200000, alpha_vals = [0.005, 0.01, 0.05], lambda_vals = [0.0005, 0.005, 0.05, 0.5]):
    for reg in lambda_vals:
        theta = L2_closed_form(X_train, y_train, reg)
        loss = test_loss(X_test, y_test, theta).item()
        print(f"Closed form loss for lambda = {reg}: {loss}")
        for step in alpha_vals:
            init_step = step
            theta = np.zeros((X_train.shape[1], 1)) 
            val_loss = 1000
            for j in range(t):
                i = np.random.randint(X_train.shape[0], size=10)
                grad = reg_gradient(X_train[i,:], y_train[i], theta).T
                theta -= step * grad
                step = step * (999/1000)
                it_val = test_loss(X_val, y_val, theta)
                if it_val > val_loss:
                    print(f'Stopped at iteration t = {j}')
                    break
                else:
                    val_loss = it_val
            loss = test_loss(X_test, y_test, theta).item()
            print(f"Initial step size: {init_step}, Test Loss: {loss}")
        print()
            # print(f"Step size: {step}, Test Loss: {loss}")

# early_stopping()
