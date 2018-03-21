#!/usr/bin/python  
# -*- coding: utf-8 -*-  

import dlib
import cv2
import time
import heapq
import numpy as np
import matplotlib.pyplot as plt
from imutils.video import VideoStream
from imutils import face_utils
import model_check
import my_utils


# --------------------- Main func ------------------------------------------
def main():
    # check existence of 68 face landmarks model
    file_name = "shape_predictor_68_face_landmarks.dat"
    model_check.model_check(file_name)
    
    # Loading model
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Start video thread
    print("[INFO] starting video stream thread...")
    webcam = 0
    video = VideoStream(webcam).start()
    time.sleep(1.0)

    # facial landmarks for the left and right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    # Mode switch: 
    # ---mode 1: initial parameters
    # ---mode 2: detection and alarm
    mode = 1
    # INITIALIZATION: open and close eye ratio
    open_eye_ratio = -1
    close_eye_ratio = 10
    open_eye_ratio_DONE = False
    close_eye_ratio_DONE = False
    # INITIALIZATION: min_heap, max_heap
    min_heap = []
    max_heap = []
    # INITIALIZATION: alarm status, closure status
    a_status = False
    c_status = False
    # INITIALIZATION: PERCLOS related
    time_gap = 5
    PERCLOS_value = 0.0
    PERCLOS_queue = []
    closure_record_queue = [0] * 101
    PERCLOS_record_queue = [0] * 101
    # INITIALIZATION: statistic graph
    plt.ion()
    title="Statistic"
    X = list(range(101))
    fig = plt.figure(title)
    # sub fig 1
    subfig1 = fig.add_subplot(211)
    subfig1.set_title("eye closure percemtage")
    line1, = subfig1.plot(X,closure_record_queue)
    axes1 = subfig1.axes
    axes1.set_ylim([0,1])
    # change y axis to percentage display
    vals = subfig1.get_yticks()
    subfig1.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])
    subfig1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    
    # sub fig2
    subfig2 = fig.add_subplot(212)
    subfig2.set_title("PERCLOS value")
    line2, = subfig2.plot(X,PERCLOS_record_queue)
    axes2 = subfig2.axes
    axes2.set_ylim([0,1])
    subfig2.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off

    # -------------------------------------------------------------------
    frame_counter = 0
    frame_counter2 = 0

    # process each frame
    while True:
        frame = video.read()
        raw_ratio = my_utils.cal_raw_ratio(frame, detector, predictor,
                                lStart, lEnd, rStart, rEnd)
        
        if raw_ratio!=-1:
            # [TEST]: output raw closure
            #print(raw_ratio)
            
            if mode == 1:
                # adjusting parameters of closure ratio
                title = "Please blink your eyes in front of the webcam"
                cv2.putText(frame, title, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                              
                heapq.heappush(min_heap, -raw_ratio) # close eye
                heapq.heappush(max_heap, raw_ratio)  # open eye
                if len(min_heap)>10:
                    heapq.heappop(min_heap)
                if len(max_heap)>10:
                    heapq.heappop(max_heap)
                
                # first 100 frames are used to initial the two heap
                if frame_counter < 100:
                    frame_counter += 1
                # find if the maximum and minimum data become stable
                else:
                    if open_eye_ratio_DONE == False:
                        open_eye_ratio_DONE, open_eye_ratio = \
                              my_utils.update_initial_ratio(-np.mean(max_heap),-open_eye_ratio)
                        open_eye_ratio = -open_eye_ratio
                        my_utils.display_status_mode1(frame,"OPEN", "PENDING")
                    else:
                        my_utils.display_status_mode1(frame,"OPEN", "DONE")
                              
                    if close_eye_ratio_DONE == False:
                        close_eye_ratio_DONE, close_eye_ratio = \
                              my_utils.update_initial_ratio(-np.mean(min_heap),close_eye_ratio)
                        my_utils.display_status_mode1(frame,"CLOSE", "PENDING")
                    else:
                        my_utils.display_status_mode1(frame,"CLOSE", "DONE")
                   
                    if (close_eye_ratio_DONE==True) and (open_eye_ratio_DONE==True):
                        mode = 2
                        frame_counter=0

            elif mode == 2:
                my_utils.display_status_mode1(frame,"OPEN", "DONE")
                my_utils.display_status_mode1(frame,"CLOSE", "DONE") 
                closure = my_utils.closure_normalization(raw_ratio,open_eye_ratio,close_eye_ratio)
                
                # PERCLOS calculation and display
                if closure > 0.8: # use P80 standard
                    c_status = True
                else:
                    c_status = False
                
                frame_counter+=1
                PERCLOS_queue.append(c_status)
                if frame_counter>=time_gap:
                    PERCLOS_value = sum(PERCLOS_queue) / float(time_gap)
                    if PERCLOS_value>0.8:
                        a_status = True
                    else:
                        a_status = False
                    frame_counter = time_gap-1
                    PERCLOS_queue.pop(0)
                
                # [WARNING] need to optimize the code here
                closure_record_queue.append(closure)
                closure_record_queue.pop(0)
                PERCLOS_record_queue.append(PERCLOS_value)
                PERCLOS_record_queue.pop(0)
                
                # Plot the statistic graph
                #my_utils.dynamic_graph(closure_record_queue,PERCLOS_record_queue)
                line1.set_ydata(closure_record_queue)
                line2.set_ydata(PERCLOS_record_queue)
                fig.canvas.draw()
                
                # Alarming phase
                frame = my_utils.alarm_system(frame,a_status)
                
                
        # show the frame
        cv2.imshow("Main frame", frame)
        
        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
    # clean up the windows and close video
    cv2.destroyAllWindows()
    video.stop()
    print("[INFO] App exited")
    
    


# --------------------- Start point ------------------------------------------
if __name__ == "__main__":
    main()



