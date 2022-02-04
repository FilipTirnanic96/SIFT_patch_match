# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:38:04 2020

@author: Filip
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time
import glob_def
if __name__ == "__main__":
    image_path = "./public/set/map.png"
    rows = []
    cols = []
    image = np.array(Image.open(image_path).convert('L'))
    #image = image[::2,1::2]
    plt.imshow(image)
    plt.show()

    for num in np.arange(0,10):
        patch_path = "./public/set/0/" + str(num) + ".png"
    
        patch = np.array(Image.open(patch_path).convert('L'))/(255 * 40 * 40)
        #patch = patch[::2,1::2]
        pw = patch.shape[1]
        pl = patch.shape[0]
        plt.imshow(patch)
        plt.show()
        cv2.imshow("image",image)
        #cv2.imshow("patch",patch)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()
        count1 = time.perf_counter()
        L1_err =  255 * np.ones((image.shape[0] - pl, image.shape[1] - pw))
        for i in np.arange(0, image.shape[0] - pw, 1): 
            for j in np.arange(0, image.shape[1] - pl, 1):
                L1_err[i,j] = np.sum(abs(image[i: i + pw, j: j+ pl] - patch))
                #print(i,j)
        count2 = time.perf_counter()       
        print('Time passed', count2 - count1)
        row, col = np.unravel_index(L1_err.argmin(), L1_err.shape)
        
        
        rows.append(row)      
        cols.append(col)
        if(num / 10 == 1):
            print(num)
        plt.imshow(image[row : row + pw, col : col + pl])
        plt.show()
        
    for r, c in zip(rows,cols):
        print(c, r)
    
    
    
    