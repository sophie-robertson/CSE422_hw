import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import yaml

# read in data
with open('newsgroup_data.yaml', 'r') as file:
    data = OrderedDict(yaml.safe_load(file)) # returns YAML data as an ordered dictionary

# data: {'newsgroup1' : {'article1' : {'word_id1' : 'count1', 'wordid_2' : 'count2', ...}, ...}, ...}
# should NOT turn earch article into length N vector
# each newsgroup is ordered and can therefore be indexed

# input: articles x, y
def jaccard_sim(x, y):
    all_words = x.keys() | y.keys() # consider only non-zero entries in both articles
    min_sum = 0
    max_sum = 0
    for word in all_words:
        x_count = x.get(word) if x.get(word) != None else 0
        y_count = y.get(word) if y.get(word) != None else 0
        min_sum += np.min(x_count, y_count)
        max_sum += np.max(x_count, y_count)
    return min_sum / max_sum

def l2_sim(x, y):
    all_words = x.keys() | y.keys() # consider only non-zero entries in both articles
    norm_sum = 0
    for word in all_words:
        x_count = x.get(word) if x.get(word) != None else 0
        y_count = y.get(word) if y.get(word) != None else 0
        norm_sum += (x_count - y_count) ** 2
    return -1 * np.sqrt(norm_sum)

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

# creates 20x20 numpy array for heatmap rendering using given similarity function
def heatmap_array(sim_func):
    result = np.empty((20,20))
    for index_1, newsgroup_1 in enumerate(data):
        for index_2, newsgroup_2 in enumerate(data):
            if (index_1 <= index_2): # decrease computation due to symmetry of heatmap
                similarity_score = 0
                num_comparisons = 0

                # compare all pairwise articles
                for (article1, bow1) in data[newsgroup_1].items():
                    for (article2, bow2) in data[newsgroup_2].items():
                        similarity_score += sim_func(bow1, bow2)
                        num_comparisons += 1
                result[index_1][index_2] = similarity_score / num_comparisons
                result[index_2][index_1] = similarity_score / num_comparisons
    return result

# creates heatmaps for all three similarity functions
def heatmap_render():
    fig, ax = plt.subplots(3)

    jaccard_arr = heatmap_array(jaccard_sim)
    ax[0] = sns.heatmap(jaccard_arr, linewidths=0.5)
    ax[0].set_title("Average Jacard Similarity across Newsgroups")

    l2_arr = heatmap_array(l2_sim)
    ax[1] = sns.heatmap(l2_arr, linewidth=0.5)
    ax[1].set_title("Average L2 Similarity across Newsgroups")

    cosine_arr = heatmap_array(cosine_sim)
    ax[2] = sns.heatmap(cosine_arr, linewidth=0.5)
    ax[2].set_title("Average Cosine Similarity across Newsgroups")

    plt.show()

if __name__ == '__main__':
    heatmap_render()
