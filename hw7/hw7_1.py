import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from PIL import Image

def get_wonderland_tree():
    # image = Image.open("data/wonderland_tree.png")
    # image = image.convert("L")
    # image_array = np.array(image)

    with open("data/wonderland_tree.txt") as file:
        lines = file.readlines()
        rows = []
        for line in lines:
            row = [int(c) for c in line if c != "\n"]
            rows.append(row)

    image_array = np.stack(rows)
    return image_array

def a():
    arr = get_wonderland_tree()
    num_ones = np.count_nonzero(arr)
    total_pixels = arr.shape[0] * arr.shape[1]
    print(arr.shape)
    print(arr)
    print(num_ones)
    print(total_pixels)
    print(f"Ratio: {num_ones / total_pixels}")
    return num_ones / total_pixels

def b(A, r, verbose = True):
    img_arr = get_wonderland_tree()
    # this is our true x
    img_vec = img_arr.flatten()
    
    A_r = A[:r, :]

    # this is b_r R^700 reconstruction of our image 
    b_r = A_r @ img_vec
    # print(b_r.shape)
    # print(A_r.shape)

    x_r = cp.Variable(shape=A_r.shape[1])
    y_r = cp.Variable(shape=A_r.shape[1])

    constraints = [A_r@x_r == b_r, 
                   y_r - x_r >= 0,
                   y_r + x_r >= 0] # x_r >= 0,  y_r >= 0
    
    obj = cp.Minimize(cp.norm(y_r, 1))

    prob = cp.Problem(obj, constraints)
    prob.solve()  
    if verbose:
        print("status:", prob.status)
        print("optimal value", prob.value)
        # print("optimal var", x_r.value) #, y_r.value)
        print(f"Close to original vector: {np.allclose(x_r.value, img_vec)}")

    return x_r.value

def c(A, verbose = True):
    img_arr = get_wonderland_tree()
    # this is our true x
    img_vec = img_arr.flatten()

    low = 0
    high = 1200
    curr_r = int((high - low) / 2) # 600
    prev_r = -1
    not_found = True
    while(not_found):
        if verbose:
            print(curr_r)
        x_r = b(A, curr_r, verbose = False)
        normed_dist = np.linalg.norm(img_vec - x_r, ord=1)

        # If we are close enough, we must test a smaller r 
        if normed_dist < 0.001:
            high = curr_r
            if verbose:
                print(f"New high: {high}")
            prev = curr_r
            curr_r = int((high - low) / 2 + low)
            if curr_r == prev:
                not_found = False
        else:
            low = curr_r
            if verbose:
                print(f"New low: {low}")
            prev = curr_r
            curr_r = int((high - low) / 2 + low)
            if curr_r == prev:
                not_found = False
    if verbose:
        print(f"High: {high}")
        print(f"Low: {low}")
        print(f"Curr r: {curr_r}")
    return curr_r

def d(A, curr_r):
    img_arr = get_wonderland_tree()
    img_vec = img_arr.flatten()

    rs = np.arange(curr_r-10, curr_r + 3)
    distances = []
    for curr_r in rs:
        x_r = b(A, curr_r, verbose = False)
        distances.append(np.linalg.norm(img_vec - x_r, ord=1))
    
    plt.figure()
    plt.plot(rs, distances)
    plt.xlabel("r in range (r-10, r-9, ..., r-1, r, r+1, r+2)")
    plt.ylabel("L-1 distance between true x and x_r")
    plt.title("Reconstruction Distance over r")
    plt.show()
    
def get_r_bar(probs):
    r_bars = []
    for prob in probs:
        stars = []
        print(prob)
        for i in range(5):

            A = np.random.choice(a = [1, -1, 0], size = (1200, 1200), p = [prob/2, prob/2, 1-prob] )
            print(np.count_nonzero(A)/ (1200*1200))
            r_star = c(A, verbose = False)
            stars.append(r_star)
        r_bars.append(np.mean(np.asarray(stars)).item())
    return r_bars
        

def e():
    print(get_r_bar([0.5]))

def f():
    # This looks wrong but I don't have time to fix it rn. 
    probs = [0.2, 0.4, 0.6, 0.8, 1.0]
    r_bars = get_r_bar(probs)
    plt.figure()
    plt.plot(probs, r_bars)
    plt.xlabel("Probability p for generation of A matrix")
    plt.ylabel("Average r* (r bar)")
    plt.title("Average r* vs. Probability for Generating A")
    plt.show()


def main():
    # a()
    # b(700)

    np.random.seed(56)
    A = np.random.normal(size=(1200, 1200))
    #c(A)
    d(A, 640)
    #e()
    # f()
    

if __name__ == '__main__':
    main()