 # -*- coding: utf-8 -*-
"""


"""

#This code is used to view the training data in visual form
#and to provide information and statistics about the training data

import numpy as np
#import time
import cv2
import matplotlib.pyplot as plt
import sys
import utilities


#np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)
#np.set_printoptions(threshold=np.nan)

#The file name of the data is provided with the command line.

if len(sys.argv) < 2:
    print("ERROR: You need to enter the file name.")
    sys.exit(1)
else:
    file_name=sys.argv[1]

tdata=np.load(file_name)

#This funtion displays the screen captures sequentially
#and information about the screen captures.
#Gives an indication that the data collection 
#is generally OK
#It accepts statistics about the data that it will display

def display_capture(tdata, stats):

    game_no=0
    
    for sample, (data, pixel_count, difference) in enumerate(zip(tdata, statistics[6], statistics[5])):
        #Since input is binary, need to multiply all 1's
        #by 255 to display
        img = data[0]*255
        #Here, uncomment to display only the upper portion of the image
        #img = img [0:80,0:134]
        #cv2.imshow('test1',img)
        #Enlarge it so it is easier to view
        img2=cv2.resize(img, (600,600),0,0,cv2.INTER_NEAREST);
        
        if not (sample in statistics[7]):
            game_no=game_no
        else:
            game_no=game_no+1    
        text1="Sample:"+str(sample)
        text2="Pixel Count:"+str(pixel_count)
        text3="Difference:"+str(difference)
        text4="Game No. "+str(game_no)
        cv2.putText(img2,text1, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img2,text2, (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img2,text3, (50,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img2,text4, (50,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.imshow('test2', img2)
        #print(choice)
        if cv2.waitKey(10) & 0xFF ==ord('q'):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break
    cv2.destroyAllWindows()

    
statistics=utilities.get_capture_statistics(tdata)
print ("Total games captured: ", statistics[0])    
print ("Size of the screen grab: ", statistics[1])
print ("Size of the key stroke grab: ", statistics[2])
print ("Total number of grabs: ", statistics[3])
print ("Total file size: ", statistics[4])
display_capture(tdata, statistics)
plt.plot(statistics[6])
plt.show()
plt.plot(statistics[5], '*')
plt.show()
#print(statistics)

cv2.destroyAllWindows()