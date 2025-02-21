import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cvxpy as cp
# from cvxpy import Variable, Minimize, Problem, multiply, tv

# Shows isolated image corruption
def no_reconstruction(Known):
    plt.gray()
    plt.imshow(Known)
    plt.show()

# Replaces unknown pixels with average of neighboring known pixels, if they exist
# Performs poorly since many unknown pixels have no neighboring known pixel, so
# it only "smooths" the edges of the corrupted line
def a(img, Known):
    unknown_indices = np.where(Known == 0)
    reconstruction = np.copy(img)
    for k in range(unknown_indices[0].size):
        i = unknown_indices[0][k]
        j = unknown_indices[1][k]
        pixel_neighbors = 0
        num_neighbors = 0
        # Neighbor above
        if i - 1 >= 0 and Known[i-1][j] != 0:
            pixel_neighbors += img[i-1][j]
            num_neighbors += 1
        # Neighbor below
        if i + 1 < img.shape[0] and Known[i+1][j] != 0:
            pixel_neighbors += img[i+1][j]
            num_neighbors += 1
        # Neighbor to left
        if j - 1 >= 0 and Known[i][j-1] != 0:
            pixel_neighbors += img[i][j-1]
            num_neighbors += 1
        # Neighbor to right
        if j + 1 < img.shape[1] and Known[i][j+1] != 0:
            pixel_neighbors += img[i][j+1]
            num_neighbors += 1
        avg_neighbors = (pixel_neighbors / num_neighbors) if num_neighbors > 0 else pixel_neighbors
        reconstruction[i][j] = avg_neighbors
    plt.gray()
    plt.imshow(reconstruction)
    plt.savefig('figures/naive_reconstruction.png')


def b(img, Known):
    U = cp.Variable(img.shape)
    obj = cp.Minimize(cp.tv(U))
    constraints = [cp.multiply(Known, U) == cp.multiply(Known, img)]
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=True)

    plt.gray()
    plt.imshow(U.value)
    plt.savefig('figures/convex_reconstruction.png')

def main():
    img = np.array(Image.open("data/jinx_corrupted.png"), dtype=int)[:,:] # reads in image
    Known = (img > 0).astype(int) # 1 if known, 0 if unknown (e.g. fully black)
    # no_reconstruction(Known)
    # a(img, Known)
    b(img, Known)

if __name__ == '__main__':
    main()


