import numpy as np


# n = number of data points, d = dimensions
def training_data(n=2000, d=20):
    X = np.random.normal(0,1, size=(n,d))
    theta_star = np.random.normal(0,1, size=(d,1))
    y = X.dot(theta_star) + np.random.normal(0,0.5,size=(n,1))
    return X, y, theta_star

# m = number of data points, d = dimensions
def test_data(m=2000, d=20):
    X = np.random.normal(0,1, size=(m,d))
    return X

def train_loss(X, y, theta):
    return np.sum((X @ theta - y) ** 2)

def test_loss(X, theta_star, theta_hat):
    return np.mean(((X @ theta_star) - (X @ theta_hat)) ** 2)

def closed_form(X, y):
    theta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    return theta_hat, train_loss(X, y, theta_hat)

def main():
    # Part (a)
    print("Part (a)")
    X_train, y_train, theta_star = training_data()
    theta_hat, irreducible_loss = closed_form(X_train, y_train)
    print(f"Irreducible loss: {irreducible_loss}")
    all_zeros_loss = train_loss(X_train, y_train, np.zeros((X_train.shape[1], 1)))
    print(f"Loss with theta = 0: {all_zeros_loss}")
    print()

    # Part (b)
    print("Part (b)")
    X_test = test_data()
    l_test = test_loss(X_test, theta_star, theta_hat)
    print(f"Average test loss for m = 2000: {l_test}")
    print()

    m_range = [10**i for i in range(2, 8)]
    for m in m_range:
        X_test = test_data(m)
        l_test = test_loss(X_test, theta_star, theta_hat)
        print(f"Average test loss for m = {m}: {l_test}")
    print()

    # Part (c)
    print("Part (c)")
    for n in range(500, 2500, 500):
        X_train, y_train, _ = training_data(n)
        theta_hat, l_train = closed_form(X_train, y_train)
        print(f"Parameter distance for n = {n}: {np.linalg.norm(theta_star - theta_hat)}")

if __name__ == '__main__':
    main()
