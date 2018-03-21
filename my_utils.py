#!/usr/bin/python  
# -*- coding: utf-8 -*-  
import time
import dlib
import cv2
from imutils import face_utils
from scipy.spatial import distance as dist
import playsound
from threading import Thread
 

def cal_raw_ratio(frame, detector, predictor,lStart, lEnd, rStart, rEnd):
    avr_closure = -1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0) 
    if len(rects)>0: 
        # find the largest face
        max_index = 0
        temp_area = 0
        for i in range(len(rects)):
            if rects[i].area()>temp_area:
                max_index = i
                temp_area = rects[i].area()
        rect = rects[max_index]
        
        # find facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # calculate closure
        leftClosure = eye_closure_ratio(leftEye)
        rightClosure = eye_closure_ratio(rightEye) 
        avr_closure = (leftClosure+rightClosure)/2.0
        
        # ----------------------------------------------------------
        # draw contours of the eyes on the screen
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (238,238,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (238,238,0), 1)
        
    return avr_closure
    
def eye_closure_ratio(eye):
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye closure ratio
    closure = (A + B) / (2.0 * C)

    return closure
        
def update_initial_ratio(temp_mean, eye_ratio):
    THRESHOLD = 0.0005
    flag = False
    if temp_mean < eye_ratio:
        if abs(eye_ratio - temp_mean) < THRESHOLD:
            flag = True
        return flag,temp_mean
    return flag,eye_ratio
    
def display_status_mode1(frame, title, status):
    width = (frame.shape)[1]
    height = (frame.shape)[0]
    color = ((50, 205, 50),(64, 64, 255))
    ind = 1
    
    if status=="DONE":
        ind = 0
        
    if title=="CLOSE":
        cv2.putText(frame, "{:<5} Ratio: ".format(title), (width-200, height-50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250, 250, 255), 2)
        cv2.putText(frame, status, (width-80, height-50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color[ind], 2)
    else:
        cv2.putText(frame, "{:<5} Ratio: ".format(title), (width-200, height-80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250, 250, 255), 2)
        cv2.putText(frame, status, (width-80, height-80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color[ind], 2)
        
def closure_normalization(raw_ratio, open_eye_ratio, close_eye_ratio):
    # closure normalization
    if raw_ratio>open_eye_ratio:
        raw_ratio = open_eye_ratio
    if raw_ratio<close_eye_ratio:
        raw_ratio = close_eye_ratio
    closure = 1 - (raw_ratio-close_eye_ratio)/(open_eye_ratio - close_eye_ratio)
    #print(closure)
    return closure

def alarm_system(frame,a_status):   
    if a_status == True:
        file_name = "alarm.wav"
        content = "WARNING!!"
        
        # Warning text for testing
        cv2.putText(frame, content, (200, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (64, 64, 255), 2)
        
        # Start Thread: sound alarm     
        
        t = Thread(target=sound_alarm, args=(file_name,))
        t.deamon = True
        t.start()
        
    return frame

def sound_alarm(file):
	playsound.playsound(file)    


        
        
        
        
        
        
        
        
        