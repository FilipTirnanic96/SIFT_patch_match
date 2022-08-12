# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:38:04 2020

@author: Filip
"""
import sys
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time
from utils.glob_def import DATA_DIR
from patch_matcher.patch_matcher import SimplePatchMatcher, AdvancePatchMatcher
from kpi_calculation.calculate_kpi import CalculateKPI 
from patch_matcher.match_visualisation import visualise_match

from IPython import get_ipython
import pandas as pd
import yaml

if __name__ == "__main__":


    #get_ipython().run_line_magic('matplotlib', 'qt')
    # get map template image
    template_image_path = os.path.join(DATA_DIR,"set","map.png")
    template = Image.open(template_image_path)
    
    
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
    flag = 1
    if flag == 1:
        # init object for kpi cals
        num_patches_to_process = 5000
        num_files = 7
        kpi_ = CalculateKPI(DATA_DIR, patch_matcher_1)
        df_kpi = kpi_.calculate_kpis(num_files, num_patches_to_process)
        accuracy = (sum(df_kpi['matched'] == 1))/df_kpi.shape[0]
        time_taken = sum(df_kpi['time'])
        print('Accuracy for n =',num_patches_to_process,'processed patches is', accuracy)
        print('Time taken for n =',num_patches_to_process,'processed patches is', time_taken)
        df_kpi.to_csv("./df_kpi_8.csv")
    elif flag == 2:
        df_kpi = pd.read_csv('./df_kpi_8.csv')
        num_visu = 20
        df_kpi_false = df_kpi[df_kpi.matched == False]
        df_kpi_false_ = df_kpi_false[df_kpi_false.n_points_matched != 0]
        df_kpi_false_['folder'] = df_kpi_false_['path'].str.slice(0,1)
        df_kpi_false_v = df_kpi_false_.iloc[-num_visu:]
        
        visualise_match(template, 'advanced', df_kpi_false_['path'], df_kpi_false_)
    else:
        patch_matcher_1 = AdvancePatchMatcher(template)