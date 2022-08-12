# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:29:03 2022

@author: uic52421
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_key_points(image, key_points):
    image_copy = image.copy()
    
    # flag if pic are 2d or 3d
    picture2d = (len(image_copy.shape) == 2)
    
    # plot keypoints
    plt.figure()
    thickness = -1
    radius = 1
    if picture2d:
        color = 0 
    else:
        color = [255, 0 ,0] 
        
    for i in np.arange(0, key_points.shape[0]):
        # get key point location
        x, y  = key_points[i]
        image_copy = cv2.circle(image_copy, (x, y), radius, color, thickness)
    
    if picture2d:
        plt.imshow(image_copy, cmap='gray', vmin=0, vmax=255) 
    else:
        plt.imshow(image_copy, vmin=0, vmax=255)
    
    #plt.show()
    
def show_matched_points(template, patch, tKP, pKP, match):
    # flag if pic are 2d or 3d
    picture2d = (len(template.shape) == 2)
    
    # extract matched key points from patch
    pt1 = pKP[match[:,1],:]
    # extract matched key points from template
    pt2 = tKP[match[:,0],:]
    
    # blank space between template img and patch img
    blank_space = 20
    # offset of patch
    offset_x = template.shape[1] + blank_space
    offset_y = int(np.round(template.shape[0]/2))
    
    if picture2d:
        merged_pic = 255 * np.ones((template.shape[0], template.shape[1] + patch.shape[1] + blank_space), dtype = int)
    
        # merge images
        merged_pic[0:template.shape[0], 0:template.shape[1]] = template
        merged_pic[offset_y:offset_y + patch.shape[0], offset_x:offset_x + patch.shape[1]] = patch
    else:
        merged_pic = 255 * np.ones((template.shape[0], template.shape[1] + patch.shape[1] + blank_space, 3), dtype = int)
        
        # merge images
        merged_pic[0:template.shape[0], 0:template.shape[1], :] = template
        merged_pic[offset_y:offset_y + patch.shape[0], offset_x:offset_x + patch.shape[1], :] = patch    
    

    # add offset to patch coordiantes
    pt1[:,0] += offset_x
    pt1[:,1] += offset_y
    
    plt.figure()
    # draw lines and key points
    thickness = 1
    radius = 4
    if picture2d:
        color = 0 
    else:
        color = [255, 0 ,0] 
        
    for i in np.arange(0, pt1.shape[0]):
        xt, yt = pt2[i]
        xp, yp = pt1[i]
        

        merged_pic = cv2.circle(merged_pic, (xt, yt), radius, color, thickness)
        merged_pic = cv2.circle(merged_pic, (xp, yp), radius, color, thickness)           
        merged_pic = cv2.line(merged_pic, (xt, yt), (xp, yp), color, thickness)
            
    if picture2d:        
        plt.imshow(merged_pic, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(merged_pic, vmin=0, vmax=255)
        
    plt.show()