import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def a(img_matrix):
    U,S,Vt = np.linalg.svd(img_matrix)
    for k in [2]:#[1, 2, 5, 10, 20, 50, 75, 100, 563]:
        U_k = U[:,0:k]
        S_k = S[0:k]
        Vt_k = Vt[0:k,:]
        k_matrix = (U_k * S_k) @ Vt_k
        print(k_matrix)

        # Pre-processing
        k_matrix_pre = k_matrix.astype(np.uint8) # float -> integer solutions
        print(k_matrix_pre)

        img_k = Image.fromarray(k_matrix)
        img_k = img_k.convert("L") # converts to grayscale after making an image

        # Post-processing matrix
        k_matrix_post = np.array(img_k)
        print(k_matrix_post)
        diff = np.nonzero(k_matrix_pre - k_matrix_post)
        print(diff)
        print(k_matrix[diff])
        print(k_matrix_pre[diff])
        print(k_matrix_post[diff])

        img_k.show()
        img_k_pre = Image.fromarray(k_matrix_pre)
        img_k_pre.show()
        # file_name = f'jinx_{k}.png'
        # img_k.save(file_name)

def c(img_matrix):
    U,S,Vt = np.linalg.svd(img_matrix)
    left = U[:,0:1]
    right = Vt[0:1,:]
    print(left[0][0])
    print(right[0][0])
        

def main():
    img = Image.open('jinx.png')
    img_matrix = np.array(img)

    print(np.mean(img_matrix[:,0])) # average luminosity of first column
    print(np.mean(img_matrix[0,:])) # average luminosity of first row
    # a(img_matrix)
    c(img_matrix)

if __name__ == '__main__':
    main()