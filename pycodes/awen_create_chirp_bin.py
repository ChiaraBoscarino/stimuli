import sys
sys.path.append('/home/awen/shared') #this is where we put all the functions.py

import numpy as np
import matplotlib.pyplot as plt

from bin_manipulation import BinFile


######################################################################################################################
#%%##################################### Parameters ####################################################################
######################################################################################################################

stimulus_dimensions = 768   #px


n_bits = 8

n_patterns = 2**n_bits

######################################################################################################################
#%%##################################### Stim generation ###############################################################
######################################################################################################################


images=np.empty((n_patterns,stimulus_dimensions,stimulus_dimensions))

for color in range(n_patterns):
    images[color::]=color



#%% TESTS


plt.imshow(images[100],cmap='gray',vmin=0,vmax=255)
plt.show()




#%% SAVING

path="/home/awen/Documents/Stims/euler/"

filename="chirp-MEA1-50Hz"




bin_file = BinFile(path+filename+".bin",                   
                   stimulus_dimensions,
                   stimulus_dimensions,
                   nb_images=n_patterns,
                   reverse=False,
                   mode='w')


images = images.astype("uint8")

for temp_image in images :
    bin_file.append(temp_image)

bin_file.close()

#%% TEST

def open_file(path, nb_images=0, frame_width=864, frame_height=864, reverse=False, mode='r'):

    file = BinFile(path, nb_images, frame_width, frame_height, reverse=reverse, mode=mode)

    return file


# bin_path='A:\Backup Documents\IDV\High def/moving_bar-150px-500x500-1000Hz.bin'
bin_path= path+filename+'.bin'
nb_images=1
bin_file = open_file(bin_path)



a= bin_file.read_frame(127)



plt.figure(figsize=(10,10))
plt.imshow(a,cmap='gray',vmin=0,vmax=1)

print(np.amax(a),np.amin(a),np.mean(a),a.shape)

bin_file.close()

