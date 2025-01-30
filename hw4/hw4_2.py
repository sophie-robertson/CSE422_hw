import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
from sklearn.decomposition import PCA


def process_pca_data():
    mapping = {"A":1, "T":2, "G":3, "C":4}
    filename = "pca-data.txt"
    bases_list = []
    pop_list = []
    pop_dict = {}
    sex_list = []
    counter = 0
    with open(filename, 'r') as file:
        for line in file:
            tokens = line.split()
            sex_list.append(int(tokens[1]))
            curr_pop = tokens[2]
            if curr_pop not in pop_dict.keys():
                pop_dict[curr_pop] = counter
                counter += 1
            pop_list.append(pop_dict[curr_pop])
            bases = tokens[3:]
            bases = [mapping[t] for t in bases]
            bases_list.append(np.asarray(bases))
    
    # n x d arr
    bases_arr = np.asarray(bases_list)
    print(bases_arr.shape)

    mode = stat.mode(bases_arr, axis = 0).mode
    bases_arr = np.where(bases_arr == mode, 1, 0)

    pop_arr = np.asarray(pop_list)
    sex_arr = np.asarray(sex_list)

    return bases_arr, pop_arr, pop_dict, sex_arr

def a(data, pop_arr, pop_dict):
    means = np.mean(data, axis=0)
    demeaned = data - means

    first_two = PCA(n_components=2)
    first_two.fit(demeaned)
    projected = first_two.transform(demeaned)

    plt.figure()

    for k in pop_dict.keys():
        l = k
        indices = np.argwhere(pop_arr == pop_dict[k])
        x = projected[indices, 0]
        y = projected[indices, 1]
        plt.scatter(x, y, label = l)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()

def c(data, pop_arr, pop_dict, sex_arr):
    means = np.mean(data, axis=0)
    demeaned = data - means

    first_three = PCA(n_components=3)
    first_three.fit(demeaned)
    projected = first_three.transform(demeaned)

    plt.figure()
    color_list= ["blue", "orange", "green",  "red", "purple", "brown", "pink"]

    for i, k in enumerate(pop_dict.keys()):
        l = k
        indices = np.argwhere(pop_arr == pop_dict[k])
        x = projected[indices, 0]
        y = projected[indices, 2]
        male = np.argwhere(sex_arr[indices] == 1)
        female = np.argwhere(sex_arr[indices] == 2)
        x_male = x[male]
        x_female = x[female]
        y_male = y[male]
        y_female = y[female]
        plt.scatter(x_male, y_male, label = f"Pop: {l}, Sex: Male", color = color_list[i], marker = '.')
        plt.scatter(x_female, y_female, label = f"Pop: {l}, Sex: Feale", color = color_list[i], marker = '*')

    plt.xlabel("PC1")
    plt.ylabel("PC3")
    plt.legend()
    plt.show()

def main():
    data, pop_arr, pop_dict, sex_arr = process_pca_data()
    # a(data, pop_arr, pop_dict)
    c(data, pop_arr, pop_dict, sex_arr)




if __name__ == '__main__':
    main()