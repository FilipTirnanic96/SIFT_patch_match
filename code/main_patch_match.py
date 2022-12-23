# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:38:04 2020

@author: Filip
"""
import os
from PIL import Image
import time
from utils.glob_def import DATA_DIR
from patch_matcher.advance_patch_matcher import AdvancePatchMatcher
from kpi_calculation.calculate_kpi import CalculateKPI 
from pach_match_visualisation.match_visualisation import visualise_match

import pandas as pd

if __name__ == "__main__":
    #get_ipython().run_line_magic('matplotlib', 'qt')
    # get map template image
    template_image_path = os.path.join(DATA_DIR,"set","map.png")
    template = Image.open(template_image_path)
    
    
    # initialise Simple Path Macher
    time_s = time.time()
    patch_matcher_1 = AdvancePatchMatcher(template)
    passt1 = time.time() - time_s
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
        model_name = "adv_pm_3_ch_ransac_public"
        file_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #file_names = [9]
        kpi_ = CalculateKPI(DATA_DIR, patch_matcher_1, model_name)
        df_kpi = kpi_.calculate_kpis_from_inputs(file_names)

    elif flag == 2:
        df_kpi = pd.read_csv(r'C:\Users\uic52421\Documents\Python Scripts\PSIML\patch_match\code\reports\adv_pm_3_ch_ransac_private_new\9.txt_n_points_matched_miss_1_less.csv')
        #df_kpi = df_kpi[df_kpi.n_points_matched >= 1]
        num_visu = 30
        df_kpi = df_kpi.iloc[:num_visu]
        
        visualise_match(template, 'advanced', df_kpi['path'], df_kpi)
    else:
        patch_matcher_1 = AdvancePatchMatcher(template)