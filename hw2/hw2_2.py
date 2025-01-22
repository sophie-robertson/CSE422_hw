# from hw2_1 import cosine_sim
import yaml
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm
from collections import OrderedDict

# read in data
with open('/Users/sophi/Downloads/UW/CSE422_hw/hw2/newsgroup_data.yaml', 'r') as file:
    full_data = OrderedDict(yaml.safe_load(file)) # returns YAML data as an ordered dictionary

# # Preprocessing to vectors
# data_dict = {}
# for index, newsgroup in enumerate(full_data):
#     data_dict[index] = {}
#     for (bow_ind, bow) in full_data[newsgroup].items():


def cosine_sim(x, y):
    all_words = x.keys() | y.keys() # consider only non-zero entries in both articles
    x_norm_sum = 0
    y_norm_sum = 0
    prod_sum = 0
    for word in all_words:
        x_count = x.get(word) if x.get(word) != None else 0
        y_count = y.get(word) if y.get(word) != None else 0
        x_norm_sum += x_count ** 2
        y_norm_sum += y_count ** 2
        prod_sum += x_count * y_count
    return prod_sum / np.sqrt(x_norm_sum * y_norm_sum)  

# Implement a baseline cosine-similarity nearest-neighbor classification system that, for any given document, finds the document with largest cosine similarity and returns the corresponding newsgroup label. You should use brute-force search.

# What is the average classification error (i.e., what fraction of the 1000 articles don’t have the same newsgroup label as their closest neighbor)?

def nearest_neighbor_search(bow1, ng_ind, bow_ind, data):
    max = -math.inf
    news_group_ind = -math.inf
    # for every newsgroup, loop through articles to find the nearest neighbor
    for index, newsgroup_2 in enumerate(data):
        # compare all pairwise articles
        for (bow2_ind, bow2) in data[newsgroup_2].items():
            # ensuring we don't check it with itself 
            if not (ng_ind == index and bow_ind == bow2_ind):
                similarity_score = cosine_sim(bow1, bow2)
                if similarity_score > max:
                    max = similarity_score
                    news_group_ind = index
    return news_group_ind

# Compute the 20 × 20 matrix whose (A, B) entry is defined by the fraction of articles in group A that have their nearest neighbor in group B. Plot the results using a heat map as in Problem 1.
def nearest_neighbor_heatmap(data):
    result = np.empty((20,20))
    error = 0
    # for each newsgroup, get the nearest neighbor for all articles
    for index_1, newsgroup_1 in enumerate(data):
        # print(data[newsgroup_1])
        print(f"Examining index {index_1}")
        counts = np.zeros(20)
        total = 0
        for (bow_ind, bow1) in data[newsgroup_1].items():
            nn_ind = nearest_neighbor_search(bow1, index_1, bow_ind, data)
            counts[nn_ind] += 1
            total += 1
        for i in range(20):
            result[index_1, i] = counts[i] / total 
        error += counts.sum() - counts[index_1]
    return result, error / 1000

# Case 1: Each entry of M is drawn randomly for a normal distribution with mean 0 and variance 1
def m1(d, N):
    return np.random.normal(loc=0.0, scale=1.0, size=(d,N))

# Case 2: Each entry of M is drawn uniformly at random from {−1, +1}.
def m2(d, N):
    return np.random.uniform(low=-1.0, high=1.0, size=(d,N))

# Case 3: Each entry of M is drawn uniformly at random from {0, 1}.
def m3(d, N):
    return np.random.uniform(low=0.0, high=1.0, size=(d,N))

def reduce_dimension(data, m):
    d = m.shape[0]
    data_copy = {}
    # for every news group
    for index, newsgroup_1 in enumerate(data):
        # create an empty dict for all articles in the newsgroup
        data_copy[newsgroup_1] = {} 
        # for every article in the newsgroup
        for (bow_ind, bow1) in data[newsgroup_1].items():
            # create empty array of length d to hold new reduced dimensional representation   
            hold = np.zeros(d)
            # get every element that has some value
            words = bow1.keys()
            # create an empty dict for new reduced dimensional representation   
            #   maps from i in {1, ..., d} --> wi*v
            data_copy[newsgroup_1][bow_ind] = {}
            # for every previous non zero word
            for word_ind in words:
                # multiply the non zero word by the element in M that would correspond to the dot product
                #   w_i * v
                #   w_i,j (i in d), (j in N) * v_j
                # We do this looping over i, and add the product for that j on to our running total
                for i in range(d):
                    hold[i] += bow1[word_ind] * m[i,word_ind]
            # Transfer everything back to the data copy 
            for dd in range(d):
                data_copy[newsgroup_1][bow_ind][dd] = hold[dd]
    return data_copy

def plot_heatmap(data):
    cosine_arr, err = nearest_neighbor_heatmap(data)
    print(f"Classification error: {err}")
    # This is about .544 -- seems high
    
    plt.imshow(cosine_arr, cmap='viridis')
    plt.xlabel("Group A")
    plt.ylabel("Group B")
    plt.text(10, 23.5, f'Classification Error: {err}', fontsize=10, ha='center', va='center')
    plt.colorbar()
    plt.show()

def plot_three(data1, data2, data3, d):
    fig, axs = plt.subplots(1, 3)
    arr_1, err1 = nearest_neighbor_heatmap(data1)
    arr_2, err2 = nearest_neighbor_heatmap(data2)
    arr_3, err3 = nearest_neighbor_heatmap(data3)
    im = axs[0].imshow(arr_1, cmap='viridis')
    axs[0].set_title(f"Heatmap when d = {d}, m = Case 1")
    axs[0].set_xlabel("Group A")
    axs[0].set_ylabel("Group B")
    axs[0].text(10, 23.5, f'Classification Error: {err1}', fontsize=10, ha='center', va='center')
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
    im2 = axs[1].imshow(arr_2, cmap='viridis')
    axs[1].set_title(f"Heatmap when d = {d}, m = Case 2")
    axs[1].set_xlabel("Group A")
    axs[1].set_ylabel("Group B")
    axs[1].text(10, 23.5, f'Classification Error: {err2}', fontsize=10, ha='center', va='center')
    plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
    im3 = axs[2].imshow(arr_3, cmap='viridis')
    axs[2].set_title(f"Heatmap when d = {d}, m = Case 3")
    axs[2].set_xlabel("Group A")
    axs[2].set_ylabel("Group B")
    axs[2].text(10, 23.5, f'Classification Error: {err3}', fontsize=10, ha='center', va='center')
    plt.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)
    plt.show()
    


if __name__ == '__main__':
    # plot_heatmap(full_data) 

    dims = [10, 30, 60, 120]
    for d in dims:
        m_1 = m1(d, 75000) 
        m_2 = m2(d, 75000)  
        m_3 = m3(d, 75000)
        data_copy_1 = reduce_dimension(full_data, m_1)
        data_copy_2 = reduce_dimension(full_data, m_2)
        data_copy_3 = reduce_dimension(full_data, m_3)
        plot_three(data_copy_1, data_copy_2, data_copy_3, d)
    print(full_data)
    # print(data)