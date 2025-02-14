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

        # Convert to image
        img_k = Image.fromarray(k_matrix)
        img_k = img_k.convert("L") # converts to grayscale after making an image
        img_k_matrix = np.array(img_k)
        file_name = f'jinx_{k}.png'
        img_k.save(file_name)

def c(img_matrix):
    U,S,Vt = np.linalg.svd(img_matrix)
    left = U[:,1:2]
    right = Vt[1:2,:]
    value = S[1]

    # get second singular vector matrix
    k_matrix = (left * value) @ right
    img_k = Image.fromarray(k_matrix)
    img_k = img_k.convert("L")
    img_k.save('jinx_k2.png')

        

def main():
    img = Image.open('jinx.png')
    img_matrix = np.array(img)

    print(np.mean(img_matrix[:,0])) # average luminosity of first column
    print(np.mean(img_matrix[0,:])) # average luminosity of first row
    # a(img_matrix)
    c(img_matrix)

if __name__ == '__main__':
    main()