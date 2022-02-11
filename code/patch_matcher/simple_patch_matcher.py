# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:17:03 2022

@author: uic52421
"""
import numpy as np
import time 
from abc import ABC, abstractmethod
import math

class PatchMatcher(ABC):
    
    def __init__(self,verbose = 0):
        self.verbose = verbose
        
    def preprocess(self, image):
        image = np.array(image.convert('L'))
        return image
    
    def first_and_second_smallest(numbers):
        m1 = m2 = float('inf')
        i1 = i2 = -1;
        for i, x in enumerate(numbers):
            if x <= m1:
                m2 = m1
                m1 = x
                i2 = i1
                i1 = i
                
            elif x < m2:
                m2 = x
                i2 = i
        return m1, i1, m2, i2

    @abstractmethod
    def extract_key_points(self, image):
        pass
    
    @abstractmethod
    def extract_features(self, key_points):
        pass
    
    @abstractmethod
    def match_patch(self, patch):
        pass    
    
    def match_features(self, patch_features, template_features, treshold):
        match = []
        # match features
        for i in np.arange(0, patch_features.shape[0]):
            patch_feature =  patch_features[i]
            
            # calculate dist from ith path_feature to each template_feature
            distance = np.sqrt(sum((template_features - patch_feature)**2))
            m1, i1, m2, i2 = self.first_and_second_smallest(distance)
            if(m1/m2 < treshold):
                match.append((i1,i))
                
        return match
            
class SimplePatchMatcher(PatchMatcher):
   
    
    def __init__(self, template_img, pw, ph, verbose = 0):
        super().__init__(verbose)
        # preprocess image
        template_img = self.preprocess(template_img)
        self.template = template_img
        self.pw = pw
        self.ph = ph
        if pw!=0:
            # extract key points from template
            self.template_key_points = self.extract_key_points(self.template)
            # extract template features
            self.template_features = self.extract_features(self.template_key_points, self.template)
        else:
            self.template_key_points = []
            self.template_features = []
        # init params
        self.time_passed_sec = -1
        self.dist = []

    
    # override abstract method
    # returns x,y keypoints location
    def extract_key_points(self, image):
        key_points_list = []
        for y in range(math.floor(self.ph/2), image.shape[0] - math.floor(self.ph/2) + 1):
            for x in range(math.floor(self.pw/2), image.shape[1] - math.floor(self.pw/2) + 1):
                key_points_list.append((x,y))
        key_points = np.array(key_points_list)
        return key_points
        
        
    # override abstract method
    # return fetures for each key point
    def extract_features(self, key_points, image):
        features = []
        for i in np.arange(0, key_points.shape[0]):
            # get key point location
            x, y  = key_points[i]
            # get patch around key point
            patch_feature = image[ y- math.floor(self.ph/2): y + math.floor(self.ph/2), x- math.floor(self.pw/2): x + math.floor(self.pw/2)]
            # ravel patch
            feature = patch_feature.ravel()
            # add feature 
            features.append(feature)
        
        features = np.array(features)
        return features
    
    # simple matcher just usees one feature for patch so it will be just one match
    def match_features(self, patch_features, template_features):
        match = []
        # match features
        for i in np.arange(0, patch_features.shape[0]):
            patch_feature =  patch_features[i]
            
            # calculate dist from ith path_feature to each template_feature
            distance = np.sqrt(np.sum((template_features - patch_feature)**2, axis = 1))
            m1, i1, m2, i2 = PatchMatcher.first_and_second_smallest(distance)
            
            self.dist = distance
            match.append((i1,i))
                
        return match
    
    # override abstract method
    # return fetures for each key point
    def match_patch(self, patch):
        # calculate time taken to match patch
        if self.verbose > 0:
            # start timer 
            start = time.time()
            
         # preprocess image
        patch = self.preprocess(patch)

        # extract key points from template
        patch_key_points = self.extract_key_points(patch)
        # extract key points from template
        patch_features = self.extract_features(patch_key_points, patch)  
        
        match = self.match_features(patch_features, self.template_features)
        i_kp_template, i_kp_patch = match[0]
        
        template_center_match = self.template_key_points[i_kp_template]
        
        x_left_top = template_center_match[0] - math.floor(self.pw/2)
        y_left_top = template_center_match[1] - math.floor(self.ph/2)
        
        if self.verbose > 0:
            # start timer 
            end = time.time()
            self.time_passed_sec = round(1.0*(end - start), 4)
            if self.verbose > 1:
                print("Time taken to match the patch", round(1.0*(end - start), 4))
                
        return x_left_top, y_left_top
    
    def preprocess(self, image):
        image = np.array(image.convert('L'))
        return image