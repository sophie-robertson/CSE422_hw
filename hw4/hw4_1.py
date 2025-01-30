import numpy as np
import matplotlib.pyplot as plt

def pca_recover(x, y):
    data = np.vstack((x, y)).T
    means = np.mean(data, axis=0)
    centered = data - means

    cov = np.matmul(centered.T, centered)  # should I switch this back?  / (x.shape[0] - 1)
    # print(cov)
    # print(np.cov(centered))
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    first_pc = eigenvectors[:, np.argmax(eigenvalues)]

    slope = first_pc[1] / first_pc[0]
    return slope

def ls_recover(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.dot(x - x_mean, y - y_mean)
    denom = np.linalg.norm(x-x_mean, 2)**2
    return num/denom

def b(x, y):
    sigmas = np.linspace(0, 0.5, 11)#[0, 0.05, 1, . . . , 0.45, 0.5]
    pcas = []
    lss = []
    for s in sigmas:
        noise = np.random.randn(1000)*s
        curr_y = np.add(y, noise)
        pcas.append(pca_recover(x, curr_y))
        lss.append(ls_recover(x, curr_y))
    
    pcas = np.asarray(pcas)
    lss = np.asarray(lss)

    plt.figure()
    plt.scatter(sigmas, pcas, color = "red", label = "PCA Recover")
    plt.scatter(sigmas, lss, color = "blue", label = "LS Recover")
    plt.xlabel("Sigma")
    plt.ylabel("Returned Slope")
    plt.legend()
    plt.show()





def main():
    # Testing helper functions
    x = np.linspace(0.001, 1, 1000) #[.001, .002, .003, ..., 1]
    y = 3 * x # [.003, .006, ... 3]

    # print(pca_recover(x, y))
    # print(ls_recover(x, y))

    b(x, y)




if __name__ == '__main__':
    main()