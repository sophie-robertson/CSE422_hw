import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def a(img_matrix):
    U,S,Vt = np.linalg.svd(img_matrix)
    for k in [10]:#[1, 2, 5, 10, 20, 50, 75, 100, 563]:
        U_k = U[:,0:k]
        S_k = S[0:k]
        Vt_k = Vt[0:k,:]
        k_matrix = (U_k * S_k) @ Vt_k
        print(k_matrix)
        k_matrix_pre = k_matrix.astype(np.uint8) # float -> integer solutions
        print(k_matrix_pre)
        img_k = Image.fromarray(k_matrix)
        k_matrix_none = np.array(img_k)
        print(k_matrix_none)
        img_k = img_k.convert("L") # converts to grayscale after making an image
        k_matrix_post = np.array(img_k)
        print(k_matrix_post)
        print(np.count_nonzero(k_matrix_pre - k_matrix_post))
        # file_name = f'jinx_{k}.png'
        # img_k.save(file_name)
        

def main():
    img = Image.open('jinx.png')
    img_matrix = np.array(img)

    a(img_matrix)

if __name__ == '__main__':
    main()