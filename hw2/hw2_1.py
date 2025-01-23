import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import yaml

# read in data
with open('hw2\\newsgroup_data.yaml', 'r') as file:
    data = OrderedDict(yaml.safe_load(file)) # returns YAML data as an ordered dictionary

# data: {'newsgroup1' : {'article1' : {'word_id1' : 'count1', 'wordid_2' : 'count2', ...}, ...}, ...}
# should NOT turn earch article into length N vector
# each newsgroup is ordered and can therefore be indexed

# input: bag of words for articles x, y
# 1 if identical
def jaccard_sim(x, y):
    all_words = x.keys() | y.keys() # consider only non-zero entries in both articles
    min_sum = 0
    max_sum = 0
    for word in all_words:
        x_count = x.get(word) if x.get(word) != None else 0
        y_count = y.get(word) if y.get(word) != None else 0
        min_sum += np.minimum(x_count, y_count)
        max_sum += np.maximum(x_count, y_count)
    return min_sum / max_sum

# 0 if identical
def l2_sim(x, y):
    all_words = x.keys() | y.keys() # consider only non-zero entries in both articles
    norm_sum = 0
    for word in all_words:
        x_count = x.get(word) if x.get(word) != None else 0
        y_count = y.get(word) if y.get(word) != None else 0
        norm_sum += (x_count - y_count) ** 2
    return -1 * np.sqrt(norm_sum)

# 1 if identical
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

def test_metrics():
    test1 = {'one': 1, "two": 1, "three": 1}
    test2 = {'one': 1, "two": 2, "four": 1}
    assert jaccard_sim(test1, test2) == 2 / 5
    assert l2_sim(test1, test2) == -1 * np.sqrt(3).item()
    assert cosine_sim(test1, test2) == 3 / (np.sqrt(3).item() * np.sqrt(6).item())

    test1 = {'one': 1, "two": 1, "three": 1, 'five': 2}
    test2 = {'one': 1, "two": 2, "four": 1}
    assert jaccard_sim(test1, test2) == 2 / 7
    assert l2_sim(test1, test2) == -1 * np.sqrt(7).item()
    assert cosine_sim(test1, test2) == 3 / (np.sqrt(7).item() * np.sqrt(6).item())
    print('Passed')

# creates 20x20 numpy array for heatmap rendering using given similarity function
def heatmap_array(sim_func):
    result = np.empty((20,20))
    for index_1, newsgroup_1 in enumerate(data):
        for index_2, newsgroup_2 in enumerate(data):
            if (index_1 <= index_2): # decrease computation due to symmetry of heatmap
                # print(index_1, index_2)
                similarity_score = 0
                num_comparisons = 0

                # compare all pairwise articles
                for (_, bow1) in data[newsgroup_1].items(): 
                    for (_, bow2) in data[newsgroup_2].items():
                        similarity_score += sim_func(bow1, bow2)
                        num_comparisons += 1
                
                # update result array with symmetry
                result[index_1][index_2] = similarity_score / num_comparisons
                result[index_2][index_1] = similarity_score / num_comparisons
    return result


# creates heatmaps for all three similarity functions
def heatmap_render():
    newsgroups = list(data.keys())
    cmap = sns.cm.rocket_r

    print('Jaccard Similarity')
    jaccard_arr = heatmap_array(jaccard_sim)
    sns.heatmap(jaccard_arr, linewidths=0.5, cmap=cmap,
                xticklabels=newsgroups, yticklabels=newsgroups) 
    plt.title("Average Jaccard Similarity across Newsgroups")
    plt.subplots_adjust(left=0.375, right=0.875, bottom=0.45, top=0.93)
    plt.show()
    plt.clf()

    print('L2 Similarity')
    l2_arr = heatmap_array(l2_sim)
    sns.heatmap(l2_arr, linewidths=0.5, cmap=cmap,
                xticklabels=newsgroups, yticklabels=newsgroups) 
    plt.title("Average L2 Similarity across Newsgroups")
    plt.subplots_adjust(left=0.375, right=0.875, bottom=0.45, top=0.93)
    plt.show()
    plt.clf()

    print('Cosine Similarity')
    cosine_arr = heatmap_array(cosine_sim)
    sns.heatmap(cosine_arr, linewidths=0.5, cmap=cmap,
                xticklabels=newsgroups, yticklabels=newsgroups) 
    plt.title("Average Cosine Similarity across Newsgroups")
    plt.subplots_adjust(left=0.375, right=0.875, bottom=0.45, top=0.93)
    plt.show()
    # # Printing all as a subplot
    # fig, axs = plt.subplots(1,3, figsize=(35,5))
    # plt.subplots_adjust(left=0.15, right=0.95, bottom=0.4, wspace=0.9)

    # print('Jaccard Similarity')
    # jaccard_arr = heatmap_array(jaccard_sim)
    # sns.heatmap(jaccard_arr, linewidths=0.5, ax=axs[0], cmap=cmap,
    #             xticklabels=newsgroups, yticklabels=newsgroups) 
    # axs[0].set_title("Average Jaccard Similarity across Newsgroups")

    # print('L2 Similarity')
    # l2_arr = heatmap_array(l2_sim)
    # sns.heatmap(l2_arr, linewidths=0.5, ax=axs[1], cmap=cmap,
    #             xticklabels=newsgroups, yticklabels=newsgroups) 
    # axs[1].set_title("Average L2 Similarity across Newsgroups")

    # print('Cosine Similarity')
    # cosine_arr = heatmap_array(cosine_sim)
    # sns.heatmap(cosine_arr, linewidths=0.5, ax=axs[2], cmap=cmap,
    #             xticklabels=newsgroups, yticklabels=newsgroups) 
    # axs[2].set_title("Average Cosine Similarity across Newsgroups")

    # plt.show()

# def heatmap_format_testing():
#     newsgroups = list(data.keys())
#     cmap = sns.cm.rocket_r
#     # fig, axs = plt.subplots(3,1, figsize=(5,50))
#     # for i in range(3):
#     uniform_data = np.ones((20,20))
#     sns.heatmap(uniform_data, linewidth=0.5, cmap=cmap,
#                 xticklabels=newsgroups, yticklabels=newsgroups)
#     plt.title(f"Plot 1")
#     #plt.subplots_adjust(left=0.25, right=1, hspace=2.5)
#     plt.subplots_adjust(left=0.375, right=0.875, bottom=0.45, top=0.93)
#     plt.show() 

def average_length():
    news_length = {}
    for index, newsgroup in enumerate(data):
        avg_length = 0
        for (_, bow) in data[newsgroup].items():
            avg_length += sum(bow.values())
        avg_length /= len(data[newsgroup])
        news_length[newsgroup] = avg_length
    print(news_length)

if __name__ == '__main__':
    # test_metrics()
    # heatmap_render()
    # heatmap_format_testing()
    average_length()
