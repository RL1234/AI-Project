in # -*- coding: utf-8 -*-
"""


"""

#This code is used to clean up the training data
#and to provide information and statistics about the training data

import numpy as np
#import time
import cv2
import sys
import utilities


#np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)
#np.set_printoptions(threshold=np.nan)

#Command line requires 1) name of original file and
#2) name of file to create

if len(sys.argv) < 3:
    print("ERROR: You need to enter 1) name of file to clean and 2) name of new file.")
    sys.exit(1)
else:
    file_name=sys.argv[1]
    file_name2=sys.argv[2]

print("'%s' will be created from '%s'." %(file_name2, file_name))

tdata=np.load(file_name)

#This function identifies the captures that have more than a certain
#total numbe of pixels (1 valued)
#This is used to identify those captures that should be excluded
#from the training set becuase they are stray or improper captures
def cleanup1(captures, threshold):
    list_of_captures=[]
    for sample,capture in enumerate(captures):
        total=np.sum(capture[0])
        #print(sample,total)
        if total>threshold:
            list_of_captures.append(sample)
    return list_of_captures

#This will identify images where there are very few or no rows where
#the total across the row is 0 or near zero
#This will indicate that the image is one where
#the aliens are moving in both x an y
#direction and should be removed

def cleanup2(captures):
    list_of_zero_rows=[]
    return_list=[]
 
    img=captures[:,0]
    for sample_no,sample in enumerate(img):
        list_of_totals=[]
        #look at only the first 20 rows, i.e., the top of the image
        for row_no, row in enumerate(sample[0:20]):
            total=np.sum(row)
            list_of_totals.append(total)
        #convert the numpy array to a list so that we can use the
        #count funtion to identify the 0 rows
        a_list=list(list_of_totals)
        numbers=a_list.count(0)
        list_of_zero_rows.append(numbers)
        
    for sample_no,total in enumerate(list_of_zero_rows):
        #Through experimentation, the threshold is 4
        #That means that any image that has less than 
        #four rows that have no pixels in them, then
        #that is the one with aliens moving in an x and y
        #direction and should be removed
        if total<4:
            return_list.append(sample_no)
        
    return return_list

#This function runs all cleanup functions and returns
#a list of the captures that should be removed
#It accepts a collection of data and a threshold used for the
#cleanup1 function
def identify_captures_to_remove(captures, total_pixel_threshold):
    a=cleanup1(captures, total_pixel_threshold)
    b=cleanup2(captures)
    #Eliminate duplicates
    for i in a:
        if i not in b:
              b.append(i)
    #Sort the list
    captures_to_remove=sorted(b)    
    return captures_to_remove
        
   
#This function rewrites the data file so that it does not include
#the frames that are in the list of captures to be removed.
#captures=image grabs
#captures to remove=list of the capture samples that should be removed
#fname=the name of the file where the data (without the excluded captures)
#should be written
def repackage_captures(captures, captures_to_remove, fname):
    new_data=[]
    for sample,capture in enumerate(captures):
        if not (sample in captures_to_remove):
            #print ("Write", sample)
            new_data.append(capture)
    np.save(fname,new_data)

problem_captures=identify_captures_to_remove(tdata, 1500)          
print("The following captures will be removed: ", problem_captures)
repackage_captures(tdata,problem_captures, file_name2)
#capture_statistics(tdata)
#repackage_captures(tdata,a, file_name3)

statistics=utilities.get_capture_statistics(tdata)
#print ("Total games captured: ", statistics[0])    
#print ("Size of the screen grab: ", statistics[1])
#print ("Size of the key stroke grab: ", statistics[2])
#print ("Total number of grabs: ", statistics[3])
#print ("Total file size: ", statistics[4])
#display_capture(tdata, statistics)
#plt.plot(statistics[6])
#plt.show()
#plt.plot(statistics[5], '*')
#plt.show()
#print(statistics)

cv2.destroyAllWindows()