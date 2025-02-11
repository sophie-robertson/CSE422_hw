import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

def load_data():
    data = np.loadtxt('data/co_occur.csv', delimiter=',') 
    with open('data/dictionary.txt') as f:
        dictionary = [line.rstrip() for line in f]
    normalized_data = np.log(data + 1)
    #print(normalized_data.shape)
    return normalized_data, dictionary

def get_top_100(data, k = 100):
    # u in mxk, s in k, vt in kxn
    # 100 columns from u, 100 rows from v
    u, s, vt = svds(data, k=k)
    # sort singular values 
    idxs = np.argsort(s)[::-1]
    u = u[:, idxs]
    s = s[idxs]
    vt = vt[idxs, :]

    return u, s, vt

def similarity(ui, vi):
    return np.dot(ui, vi) 

def a(s):
    plt.figure()
    x_axis = np.arange(1, 101)
    plt.scatter(x_axis, s)
    plt.xlabel("Singular Value Number")
    plt.ylabel("Value")
    plt.title("Top 100 Singular Values for M hat")
    plt.show()

def b(u, dictionary):
    for j in range(u.shape[1]):
        print(j)
        uj = u[:,j]
        sorted = np.argsort(uj)[::-1]
        biggest = sorted[0:10]
        smallest = sorted[-10:]
        words_biggest = [dictionary[b] for b in biggest]
        words_smallest = [dictionary[s] for s in smallest]
        print(words_biggest)
        print(words_smallest)
        print("\n")

def c_d(u, dictionary, words):
    plt.figure()

    norms = np.linalg.norm(u, axis=1, keepdims=True)  
    norm_u = u / norms  
    woman_ind = dictionary.index("woman")
    man_ind = dictionary.index("man")
    difference = norm_u[woman_ind] - norm_u[man_ind]

    projs = []
    for i, w in enumerate(words):
        ind = dictionary.index(w)
        v_vect = norm_u[ind]
        proj = np.dot(v_vect, difference)
        projs.append(proj)
        col = ""
        if proj < 0:
            col = "blue"
        else:
            col = "pink"
        plt.bar(i, proj, label = w, color = col)

    plt.title("Projections Onto 'woman' - 'man' Embedding")
    plt.xticks(range(len(words)), words)
    plt.show()

import bisect

def e1(u, dictionary):
    norms = np.linalg.norm(u, axis=1, keepdims=True)  
    norm_u = u / norms  
    target = 'washington'
    embedding = norm_u[dictionary.index(target)]
    top_10_words = []
    top_10_sim = []
    for i in range(len(dictionary)):
        # if (i != dictionary.index(target)):
        sim = similarity(embedding, norm_u[i])

        # reverse list (low to high similarity)
        top_10_sim.reverse()
        top_10_words.reverse()
        # insert word at correct index
        index = bisect.bisect_right(top_10_sim, sim)
        top_10_words.insert(index, dictionary[i])
        top_10_sim.insert(index, sim)
        # reverse list (high to low similarity)
        top_10_sim.reverse()
        top_10_words.reverse()

        # take only top 11 words (including washington)
        top_10_words = top_10_words[:11]
        top_10_sim = top_10_sim[:11]
    return (top_10_words, top_10_sim)

def cosine_similarity_vectorized(x, u):
    # x in n x d
    # y in m x d
    # y.T in d x m
    print(x.shape)
    print(u.shape)
    dotted = x @ u.T
    # dotted in n x m
    # norms in n x m
        # at x[ij], divide by x_norm[i] * y_norm [j]
    # x_norm in n
    x_norm = np.linalg.norm(x, ord = 2, axis=1, keepdims = True)
    # y_norm in m
    y_norm = np.linalg.norm(u, ord = 2, axis=1, keepdims = True)

    outer = np.outer(x_norm.squeeze(), y_norm.squeeze())

    return dotted / (outer + 1e-8)

def load_embedded_analogies(norm_u, dictionary):
    analogies = []
    labels = []
    with open('data/analogy_task.txt') as f:
        for line in f:
            tokens = line.rstrip().split()
            a = tokens[0]
            b = tokens[1]
            aa = tokens[2]
            bb = tokens[3]
            target_vector = norm_u[dictionary.index(b)] - norm_u[dictionary.index(a)] + norm_u[dictionary.index(aa)]
            analogies.append(target_vector)
            labels.append(bb)

    # n analogies x vector length (100)
    analogies = np.stack(analogies)
    print(analogies.shape)
    print(len(labels))
    return analogies, labels




def e2(u, dictionary):
    norms = np.linalg.norm(u, axis=1, keepdims=True)  
    norm_u = u / norms  

    analogies, labels = load_embedded_analogies(norm_u, dictionary)

    # should be n x m
    similarities = cosine_similarity_vectorized(analogies, norm_u)
    print(similarities.shape)

    indices = np.argmax(similarities, axis=1)
    print(indices.shape)

    num_correct = 0
    for i in range(len(indices)):
        prediction = dictionary[indices[i]]
        print(f"Prediction: {prediction}, Correct: {labels[i]}")
        if prediction == labels[i]:
            num_correct += 1


    return num_correct/len(indices)





def main():
    print("Getting SVD")
    data, dictionary = load_data()
    u, s, vt = get_top_100(data)

    # a(s)

    # b(u, dictionary)

    # c_words = ["boy", "girl", "brother", "sister", "king", "queen", "he", "she", "john", "mary", "all", "tree"]
    # c_d(u, dictionary, c_words)

    # d_words = ["math", "matrix", "history", "nurse", "doctor", "pilot", "teacher", "engineer", "science", "arts", "literature", "bob", "alice"]
    # c_d(u, dictionary, d_words)
    print('Starting Analysis')
    print(e1(u, dictionary))

    # print(f"accuracy = {e2(u, dictionary)}")

if __name__ == '__main__':
    main()