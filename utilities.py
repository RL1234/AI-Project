# -*- coding: utf-8 -*-
"""

"""

#This function accepts a group of screen grabs and returns statistics    
import numpy as np

def get_capture_statistics(captures):
    total_games_captured=0
    last_total=int(0)
    images=captures[0]
    keystrokes=captures[1]
    pixel_totals=[]
    pixel_difference=[]
    #last_image=tdata[0,0]
    last_image=images[0]
    last_image=last_image[0:80,0:134]
    game_start_sample=[]    

    screen_grab_size=images[0].size
    keystrokes_grab_size=keystrokes[1].size
    total_grabs=len(captures)    
    file_size=len(captures)*(images[0].size+keystrokes[1].size)
    for sample,capture in enumerate(captures):        
        total=int(np.sum(capture[0]))
        difference=abs(total-last_total)
        #print (sample, last_total, total, difference)
        if difference>800:
            #print("New Game", sample)
            total_games_captured=total_games_captured+1
            game_start_sample.append(sample)
        last_total=total
    
    for data in captures[1:]:
            img = data[0]
            #deal only with the pixels in the upper part of the image
            #becuase it is constantly changing and is the focus
            #of determining these statistics.
            img = img[0:80,0:134]
            difference=np.absolute(img-last_image)
            difference=np.sum(difference)
            difference=difference/(np.sum(last_image))*0.5
            total=np.sum(img)
            pixel_totals.append(total)
            pixel_difference.append(difference)
            last_image=img

    return total_games_captured, screen_grab_size,\
    keystrokes_grab_size, total_grabs, file_size,\
    pixel_difference,pixel_totals, game_start_sample