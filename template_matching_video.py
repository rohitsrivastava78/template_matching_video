#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:21:54 2019

@author: rohit
"""

import cv2
import numpy as np


template_img_path='template.jpg'
template_img=cv2.imread(template_img_path)
template_img=cv2.cvtColor(template_img,cv2.COLOR_BGR2GRAY)
w,h=template_img.shape[::-1]
    
source_video='led1.mp4';#source_video=0;
cap = cv2.VideoCapture(source_video)
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
     
# Read until video is completed
main_loop=1
while(main_loop==1 and cap.isOpened() ):
    # Capture frame-by-frame
    ret, img = cap.read()
    if ret == True:
     
        # Display the resulting frame
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
    
        match_img=cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
        CONFIDENCE=0.7
        loc=np.where( match_img > CONFIDENCE)
     
        #*loc[::-1] -> extract individual element in tupple & swap x, y coord 
        max_confidence=0; final_pt=None; loc_points=[]; pt_prev=(0,0); PT_THRESH=50;
        if len(loc[:][0]) == 0:
            continue
        loc_points=np.concatenate( (np.reshape(loc[::-1][0],(len(loc[:][0]),1)), 
                                  np.reshape(loc[::-1][1],(len(loc[:][0]),1))),axis=1 )
            
        for pt in loc_points:
            #print("point value : {}, confidence : {}".format(pt, match_img[pt[1],pt[0]]))
            confidence=match_img[pt[1],pt[0]];
            final_pt=pt

            print("final point value : {}, confidence : {}".format(final_pt, match_img[final_pt[1],final_pt[0]]))
            cv2.rectangle(img, (final_pt[0], final_pt[1]), (final_pt[0]+w, final_pt[1]+h), (0,0,255),1)
            cv2.imshow('matched_img',img)
        #cv2.waitKey(100)        
        k = cv2.waitKey(500)
        if k == 27:
            break
        
    else:
        break
# When everything done, release the video capture object
cap.release() 
# Closes all the frames
cv2.destroyAllWindows()