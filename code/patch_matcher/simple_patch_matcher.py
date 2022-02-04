# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:17:03 2022

@author: uic52421
"""
import numpy as np
import time 

class SimplePatchMatcher:
    
    def __init__(self, template_img, verbose = 0):
        # preprocess image
        template_img = self.preprocess(template_img)
        # init class params
        self.template = template_img
        self.verbose = verbose
        self.time_passed_sec = -1
        
    def match_patch(self, patch):
        # preprocess patch
        patch = self.preprocess(patch)
        # template shape
        h, w = self.template.shape
        # patch shape
        ph, pw = patch.shape
        # Loss fcn
        l = np.Inf 
        # left top position of match patch
        x_match = -1
        y_match = -1
        
        if self.verbose > 0:
            # start timer 
            start = time.time()
            
        # convolve template with patch
        for x in np.arange(0, w - pw):
            for y in np.arange(0, h- ph):
                loss = np.sum(abs(self.template[y: y + ph, x: x+ pw] - patch))
                if loss < l:
                    l = loss
                    x_match = x
                    y_match = y
        
        if self.verbose > 0:
            # start timer 
            end = time.time()
            self.time_passed_sec = round(1.0*(end - start), 4)
            if self.verbose > 1:
                print("Time taken to match the patch", round(1.0*(end - start), 4))
            
        return x_match, y_match
    
    def preprocess(self, image):
        image = np.array(image.convert('L'))
        return image