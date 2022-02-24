# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:38:04 2020

@author: Filip
"""
import sys
import os
if './' not in sys.path:
    sys.path.append('./')
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time
from utils.glob_def import DATA_DIR
from patch_matcher.patch_matcher import SimplePatchMatcher, AdvancePatchMatcher
from kpi_calculation.calculate_kpi import CalculateKPI 

from IPython import get_ipython

def visualize_patch(template, patch, x, y, ph, pw):
    template_copy = template.copy()
    cv2.rectangle(template_copy, (x, y), (x + pw, y + ph), (255,0,0), 3)
    plt.imshow(np.array(template_copy))
    plt.show() 
    plt.imshow(np.array(patch))
    plt.show() 

if __name__ == "__main__":
    #get_ipython().run_line_magic('matplotlib', 'qt')
    # get map template image
    template_image_path = os.path.join(DATA_DIR,"set","map.png")
    template = Image.open(template_image_path)
    
    # show template
    plt.imshow(np.array(template))
    plt.show()
    
    
    # initialise Simple Path Macher
    time_s = time.time()
    patch_matcher_1 = AdvancePatchMatcher(template)
    passt1 =  time.time() - time_s
    print(passt1)
    # take first n patches
    n_patches = 10
    # cumulative time taken
    t_cum = 0
    
    # initialise Simple Path Macher
    '''time_s = time.time()
    patch_matcher_ = SimplePatchMatcher(template, 40, 40, 2)
    passt2 =  time.time() - time_s 
    print(passt2)
    '''
    '''
    # init object for kpi cals
    num_patches_to_process = 20
    kpi_ = CalculateKPI(DATA_DIR, patch_matcher_)
    df_kpi = kpi_.calculate_kpis(-1, num_patches_to_process)
    accuracy = (sum(df_kpi['matched'] == 1))/df_kpi.shape[0]
    time_taken = sum(df_kpi['time'])
    print('Accuracy for n =',num_patches_to_process,'processed patches is', accuracy)
    print('Time taken for n =',num_patches_to_process,'processed patches is', time_taken)
    '''
    # init object for kpi cals
    num_patches_to_process = 3000
    kpi_ = CalculateKPI(DATA_DIR, patch_matcher_1)
    df_kpi = kpi_.calculate_kpis(4, num_patches_to_process)
    accuracy = (sum(df_kpi['matched'] == 1))/df_kpi.shape[0]
    time_taken = sum(df_kpi['time'])
    print('Accuracy for n =',num_patches_to_process,'processed patches is', accuracy)
    print('Time taken for n =',num_patches_to_process,'processed patches is', time_taken)