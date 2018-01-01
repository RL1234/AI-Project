# -*- coding: utf-8 -*-
"""


"""

#This code is used to grab images of the game
#as it is being played and stores them in a numpy array
#It also captures the user's input during game play

import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
import win32api as wapi
import os
import sys

#Set the coordinates to grab
screen_grab_coordinates=(0,150,670,700)

#Set name of file where training data will be stored
#file_name = 'training_data.npy'

if len(sys.argv) < 2:
    print("ERROR: You need to enter the file name where training data should be saved.")
    sys.exit(1)
else:
    file_name=sys.argv[1]

tdata=np.load(file_name)


#A timer to provide time to orient the screen before the screen grabs begin
def countdown(countdown_time):
    count_to=countdown_time
    while(count_to!=0):
        time.sleep(1)
        count_to=count_to-1
        print (count_to)

#This repeatedly grabs and displays the screen
def screen_record(coordinates): 

    last_time=time.time()  
    printscreen =  np.array(ImageGrab.grab(bbox=coordinates))
    newscreen = process_img(printscreen)
    #the image is binary 0 and 1, so need to multiply by 255 so all the 
    #1s become 255 and the image can be shown with imshow
    cv2.imshow('window',newscreen*255)
    #move the display window away from the
    #game window
    cv2.moveWindow('window', 820, 0)
    print ("Screen Grab Time:", time.time()-last_time)
    last_time=time.time()
    return newscreen

#This converts the grabbed screen to show only lines by edge detection
#and then reduces the image
def process_img(image):
    
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # edge detection
    #processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    
    #Convert to binary 0s and 1s by making every value over 60 a 1.
    #All the 0s stay 0
    #ret, processed_img = cv2.threshold(processed_img,60,1,cv2.THRESH_BINARY)    
    ret, processed_img = cv2.threshold(processed_img,100,1,cv2.THRESH_BINARY)    
    
    #reduce the screen/resolution by a factor or 5 to decrease the amount of data
    #No need for so much data for a black and white image
    height, width = processed_img.shape[:2]
    new_height = int(height/5)
    new_width = int(width/5)
    processed_img = cv2.resize(processed_img, (new_width,new_height))

    return processed_img

#This gets the key being pressed [or breaks on esc]
def get_keystroke():
    #right arrow
    if(wapi.GetAsyncKeyState(0x27)):
        return np.array([0,0,1,0], dtype=np.uint8)
    #left arrow    
    elif(wapi.GetAsyncKeyState(0x25)):
        return np.array([1,0,0,0], dtype=np.uint8)
    #space bar
    elif(wapi.GetAsyncKeyState(0x20)):
        return np.array([0,1,0,0],dtype=np.uint8)
#    elif(wapi.GetAsyncKeyState(0x1B)):
#        return 1
    #No key press
    else:
        return np.array([0,0,0,1], dtype=np.uint8)

#writes the image (i) and key strokes (k) to the list (data) and then
#adds both to the file name (fname) every n grabs        
def write_data(data, i, k, fname, n):
    #Since the data is binary, we need only the last bit of each byte
    #So we use packbits.  This reduces data by a factor of 8
    #When packing the image, need to set axis=1 to preserve shape
    #of the data.  Otherwise, the function flattens the data set by default
    #i = np.packbits(i, axis=1)
    #k = np.packbits(k)
    
    data.append([i,k])
    if len(data) % n == 0:
        print(len(data))
        np.save(fname,data)

        
#This isues commands to the game, i.e., go left, right or fire
#Not used
def do_something():
    pyautogui.press('left')
    print("Moved Left")
    pyautogui.press('space')
    print("Fired")
    
def open_training_data_file(fname):
    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        training_data = list(np.load(file_name))
    else:
        print('File does not exist, starting fresh!')
        training_data = []

    return training_data
    
       
#MAIN    
#Open existing training data if it exists and prepare to add more training data
tdata=open_training_data_file(file_name)


#Count a certain number of seconds before starting the screen grab    
countdown(2)        
#Continuously perform screen grabbing of the specified coordinates


while True:
    image=screen_record(screen_grab_coordinates)
    k=get_keystroke()
    print (k)
    #do_something()
    #Uncomment to save data     
    write_data(tdata, image, k, file_name, 10)        

        #Stop when user presses 'q'
    if cv2.waitKey(250) & 0xFF == ord('q'):
        cv2.destroyAllWindows() 
        break



 
                                