# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:17:03 2022

@author: uic52421
"""
import numpy as np
import time 
from abc import ABC, abstractmethod
import math
from scipy import signal, ndimage
import cv2
import scipy.ndimage.filters as filters

import matplotlib.pyplot as plt

def debug_image(image):
    image_copy = image.copy().astype('float64')
    image_copy /= image_copy.max()/255.0 
    plt.imshow(image_copy)
    plt.show()
    
def show_key_points(image, key_points):
    image_copy = image.copy()
    ps = 1
    for i in np.arange(0, key_points.shape[0]):
        # get key point location
        x, y  = key_points[i]
        image_copy[y-ps:y+ps,x-ps:x+ps] = 0
        #image_copy[y,x] = 0
    plt.imshow(image_copy, cmap='gray', vmin=0, vmax=255)
    plt.show()
    
class PatchMatcher(ABC):
    
    def __init__(self,verbose = 0):
        self.verbose = verbose
        self.time_passed_sec = -1
        
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
    
    def match_features(self, patch_features, template_features):
        match = []
        treshold = 0.7
        # match features
        for i in np.arange(0, patch_features.shape[0]):
            patch_feature =  patch_features[i]
            
            # calculate dist from ith path_feature to each template_feature
            distance = np.sqrt(np.sum((template_features - patch_feature)**2, axis = 1))
            m1, i1, m2, i2 = PatchMatcher.first_and_second_smallest(distance)
            if(m1/m2 < treshold):
                match.append((i1,i))
                
        return match
    
    @abstractmethod
    def find_correspodind_location_of_patch(self, patch_key_points, match):
        pass
    
    # returns left top location on template image of matched patch
    def match_patch(self, patch):
        # calculate time taken to match patch
        start = time.time()
            
         # preprocess image
        patch = self.preprocess(patch)

        # extract key points from template
        patch_key_points = self.extract_key_points(patch)
        # extract key points from template
        patch_features = self.extract_features(patch_key_points, patch)  
        # find feature matchs between patch and template
        match = self.match_features(patch_features, self.template_features)
        # find top left location on template of matched patch
        x_left_top, y_left_top = self.find_correspodind_location_of_patch(patch_key_points, match)
        
        end = time.time()
        self.time_passed_sec = round(1.0*(end - start), 4)
        if self.verbose > 0:
            print("Time taken to match the patch", round(1.0*(end - start), 4))
                
        return x_left_top, y_left_top  
    
    
            
class SimplePatchMatcher(PatchMatcher):
   
    
    def __init__(self, template_img, pw, ph, verbose = 0):
        super().__init__(verbose)
        # preprocess image
        template_img = self.preprocess(template_img)
        self.template = template_img
        self.pw = pw
        self.ph = ph
        if (pw!=0 and ph!=0):
            # extract key points from template
            self.template_key_points = self.extract_key_points(self.template)
            # extract template features
            self.template_features = self.extract_features(self.template_key_points, self.template)
        else:
            self.template_key_points = []
            self.template_features = []
    
    # override abstract method
    # every point in image is key point
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
    
    # simple matcher just uses one feature for patch so it will be just one match
    # outputs list of meatched features
    def match_features(self, patch_features, template_features):
        match = []
        # match features
        for i in np.arange(0, patch_features.shape[0]):
            patch_feature =  patch_features[i]
            
            # calculate dist from ith path_feature to each template_feature
            distance = np.sqrt(np.sum((template_features - patch_feature)**2, axis = 1))
            m1, i1, m2, i2 = PatchMatcher.first_and_second_smallest(distance)
            
            
            match.append((i1,i))
                
        return match
    
    # override abstract method
    # output top let postion of template where the patch match
    def find_correspodind_location_of_patch(self, patch_key_points, match):
        i_kp_template, i_kp_patch = match[0]
        
        template_center_match = self.template_key_points[i_kp_template]
        
        x_left_top = template_center_match[0] - math.floor(self.pw/2)
        y_left_top = template_center_match[1] - math.floor(self.ph/2)
        return x_left_top, y_left_top
 
    
  
class AdvancePatchMatcher(PatchMatcher):
    
    def __init__(self, template_img, verbose = 0):
        super().__init__(verbose)
        # init params
        self.grad_mag = []
        self.grad_theta = []
        # preprocess image
        template_img = self.preprocess(template_img)
        self.template = template_img
        # extract key points from template
        self.template_key_points = self.extract_key_points(self.template)
        # extract template features
        self.template_features = self.extract_features(self.template_key_points, self.template)
    
    # override abstract method
    # use Harris detector to extrack corner points
    # returns x,y keypoints location
    def extract_key_points(self, image):
        key_points_list = []
        
        # find image gradiants
        # % gradient image, for gradients in x direction.
        img_dx = 1.0*np.absolute(signal.convolve2d(image, np.reshape(np.array([-1, 0, 1]), (1,-1)), mode='same', boundary = 'symm'))/255 
        # % gradients in y direction.
        img_dy = 1.0*np.absolute(signal.convolve2d(image, np.reshape(np.array([-1, 0, 1]), (-1,1)), mode='same', boundary = 'symm'))/255 
        
        # calculate values for M matrix
        img_dx2 = 1.0*img_dx**2
        img_dy2 = 1.0*img_dy**2
        img_dxy = 1.0*img_dx * img_dy
        
        self.grad_mag = img_dx2 + img_dy2
        self.grad_theta = np.arctan2(img_dy, img_dx) 
        # blur gradiants
        img_dx2 = ndimage.gaussian_filter(img_dx2, sigma = 2, truncate = 1)
        img_dy2 = ndimage.gaussian_filter(img_dy2, sigma = 2, truncate = 1)
        img_dxy = ndimage.gaussian_filter(img_dxy, sigma = 2, truncate = 1)
        
        # calculate det and trace for finding R
        detA = (img_dx2 * img_dy2) - (img_dxy**2)
        traceA = (img_dx2 + img_dy2)
        
        # calculate response for Harris Corner equation
        k = 0.05
        R = detA - k*(traceA**2)
        # find key points where R > threshold
        threshold = 0.0001*R.max()
        R[R < threshold] = 0
        # non maxima supresion
        R_max = filters.maximum_filter(R, 3)
        R[R != R_max] = 0
        # extract key points
        key_points_indeces = np.where(R > 0)
        key_points_list = [(key_points_indeces[1][i], key_points_indeces[0][i]) for i in np.arange(0, key_points_indeces[0].shape[0])]

        
        key_points = np.array(key_points_list)
        #show_key_points(image, key_points)
        return key_points
    
    def compute_gradient_histogram(self, num_bins, gradient_magnitudes, gradient_angles):
        angle_step = 2 * np.pi / num_bins
        angles = np.arange(0, 2*np.pi + angle_step, angle_step)
    

        indices = np.digitize(gradient_angles.ravel(), bins = angles)
        indices -= 1 
        gradient_magnitudes_ravel = gradient_magnitudes.ravel();
        histogram = np.zeros((num_bins));
        for i in range(0, indices.shape[0]):
           histogram[indices[i]] +=  gradient_magnitudes_ravel[i]
        
        return histogram
        
    #
    def extract_features(self, key_points, image):
        features = []
        patch_size = 5
        num_angles = 8
        
        for i in np.arange(0, key_points.shape[0]):
            # get key point location
            x, y  = key_points[i]
            # get patch around key point
            min_y = max(0, y- math.floor(patch_size/2))
            max_y = min(y + math.floor(patch_size/2) + 1, image.shape[0] - 1)
            min_x = max(0, x- math.floor(patch_size/2))
            max_x = min(x + math.floor(patch_size/2) + 1, image.shape[0] - 1)
            
            patch_grad = self.grad_mag[ min_y: max_y, min_x: max_x]
            patch_theta = self.grad_theta[ min_y: max_y, min_x: max_x]
            
            feature = self.compute_gradient_histogram(num_angles, patch_grad, patch_theta)
            features.append(feature) 
        
        features = np.array(features)
        return features
    
    def find_correspodind_location_of_patch(self, patch_key_points, match):
        return 0,0