# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:44:17 2022

@author: uic52421
"""
import os
from PIL import Image
import numpy as np
import time
import pandas as pd

class CalculateKPI:
    
    
    def __init__(self, path_to_dataset):
        self.dataset_path = path_to_dataset
        self.path_to_input_txt_files = os.path.join(path_to_dataset, 'inputs')
        self.path_to_onput_txt_files = os.path.join(path_to_dataset, 'outputs')
        self.path_to_input_patches = os.path.join(path_to_dataset, 'set')
        # init patch matcher
        # patch_matcher_ = PatchMatcher()
        
    def calculate_kpis(self, num_files = -1, num_inputs = -1):
        # proces template image
        template_image_path = os.path.join(self.dataset_path,"set","map.png")
        template = Image.open(template_image_path)
        # extract features for template
        # template_features = patch_matcher_.extract_features(template_features)
        template_ = np.array(template.convert('L'))
        # init params
        processed_files = 0
        processed_inputs = 0
        # init list to store kpis
        kpis = []
        
        # excided num inputs flag
        excided_num_inputs = False
        
        # read each txt file
        for txt_file in os.listdir(self.path_to_input_txt_files):
            # open ground truth file
            output_f = open(os.path.join(self.path_to_onput_txt_files,txt_file), 'r')
            
            # open cuurent input.txt file
            with open(os.path.join(self.path_to_input_txt_files,txt_file), 'r') as f:
                # read initial params
                path_to_template = f.readline()
                num_patches = int(f.readline())
                patch_size_str = f.readline().split()
                ph = int(patch_size_str[0])
                pw = int(patch_size_str[1])
                
                # read path to each patch
                path_to_patch = f.readline()
                while path_to_patch:
                    # init kpi list for current patch 
                    kpi_list = []
                    # take relative path from dataset path
                    path_to_patch = path_to_patch.split('/')
                    path_to_patch = os.path.join(path_to_patch[1], path_to_patch[2])
                    # delete new line read from input
                    path_to_patch = path_to_patch[:-1]
                    
                    # append relative to list
                    kpi_list.append(path_to_patch)
                    
                    # make full path to patch
                    path_to_patch = os.path.join(self.path_to_input_patches, path_to_patch)
                    # calculate time taken to match a patch
                    time_start = time.time()
                    
                    # read patch
                    patch = Image.open(path_to_patch)
                    # extract features from patch
                    # patch_features = patch_matcher_.extract_features(patch)
                    patch_ = np.array(patch.convert('L'))
                    # match patch to template
                    # [x_match, y_match] = patch_matcher_.match(template_features, patchfeatures)
                    x_match = 0
                    y_match = 0
                    
                    # append x_match, y_match to list
                    kpi_list.append(x_match)
                    kpi_list.append(y_match)
                    
                    # read expected output
                    expected_output = output_f.readline().split()
                    x_expected = int(expected_output[0])
                    y_expected = int(expected_output[1])

                    # append x_expected, y_expected to list
                    kpi_list.append(x_expected)
                    kpi_list.append(y_expected)
                    
                    # check if we are in 120 neiborhood (6x6 pixels missed)
                    matched = False
                    if (abs(x_expected - x_match) <= 5 and abs(y_expected - y_match) <= 5):
                        matched = True
                    
                    time_end = time.time()
                    
                    # append matched to list
                    kpi_list.append(matched)
                    
                    # append time taken to list
                    kpi_list.append(round(1.0*(time_end - time_start), 4))
                    
                    # increment processed measurement
                    processed_inputs = processed_inputs + 1                    
                    
                    # if we proccess defined number of patches break
                    if((num_inputs!= -1) and (processed_inputs >= num_inputs)):
                        excided_num_inputs = True
                        f.close()
                        output_f.close()
                        break
                    
                    # store kpi_list to list of all kpis
                    kpis.append(kpi_list)
                    # read new patch path
                    path_to_patch = f.readline()
                    
            # increment processed file
            processed_files = processed_files + 1
            
            # if we break from excided_num_inputs break from outer for
            if excided_num_inputs:
                break
                
            # if we proccess defined number of patches break
            if((num_inputs!= -1) and (processed_files >= num_files)):
                f.close()
                output_f.close()
                break
            
            # close current output file
            output_f.close()
            
        df_kpi = pd.DataFrame(kpis, columns = ['path','x_match', 'y_match', 'x_expected', 'y_expected', 'matched', 'time'])
        return df_kpi