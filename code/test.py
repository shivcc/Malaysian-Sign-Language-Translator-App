#this code will convert the csv files into flatened np array for the x input of the model

import csv
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical



#This code will be used in the training model code for reading data
# standardize video size ie 3 sec 90frames


def dat_pose_extractor(csv_file):
    
    #opens the csv data
    csvfile = open(csv_file , newline='')
    spamreader = csv.reader(csvfile)
    pose_data = np.array(np.zeros((90,4*33)))
    pose1 = np.array([])
    pose2 = np.array([])
    i = 0
    for row in spamreader:

     if i!= 0 and row != ['x','x','x','x'] and row != ['E','E','E','E']:
          pose1 = np.append(pose1, row) 
     elif row == ['E','E','E','E']:
         for s in range(33):
             pose1 = np.append(pose1, ['0','0','0','0']) 
     elif row == ['x','x','x','x']:  #end of an image 
         pose2 = np.append(pose2, pose1) # pose1 fully loaded with image
         pose1 = np.array([])   
     i+=1
    
    pose2 = pose2.reshape(int(len(pose2)/132),4*33) #from 6864 to (vid_frames,image_data_132)
    for w in range(len(pose2)):
        pose_data[w]= pose2[w]
    
    csvfile.close()
    return pose_data
 
def dat_lh_extractor(csv_file):
    csvfile = open(csv_file , newline='')    
    spamreader = csv.reader(csvfile)
    lh_data = np.array(np.zeros((90,3*21)))
    pose1 = np.array([])
    pose2 = np.array([])
    i = 0
    for row in spamreader:
        
     if i!= 0 and row != ['x','x','x'] and row != ['E','E','E']:
          pose1 = np.append(pose1, row) 
     elif row == ['E','E','E']:
          for s in range(21):
              pose1 = np.append(pose1, ['0','0','0']) 
     elif row == ['x','x','x']:   
         pose2 = np.append(pose2, pose1)
         pose1 = np.array([])
        
     i+=1
    
    pose2 = pose2.reshape(int(len(pose2)/63),3*21) 
    for w in range(len(pose2)):
        lh_data[w]= pose2[w]
        
    csvfile.close()
    return lh_data


def dat_rh_extractor(csv_file):
    csvfile = open(csv_file , newline='')
    spamreader = csv.reader(csvfile)
    rh_data = np.array(np.zeros((90,3*21)))
    pose1 = np.array([])
    pose2 = np.array([])
    i = 0
    for row in spamreader:
        
     if i!= 0 and row != ['x','x','x'] and row != ['E','E','E']:
          pose1 = np.append(pose1, row) 
     elif row == ['E','E','E']:
          for s in range(21):
              pose1 = np.append(pose1, ['0','0','0'])    
     elif row == ['x','x','x']:  #dont foget to make the split the array to seperate images 
         pose2 = np.append(pose2, pose1)
         pose1 = np.array([])
     i+=1
    
    pose2 = pose2.reshape(int(len(pose2)/63),3*21) 
    for w in range(len(pose2)):
        rh_data[w]= pose2[w]
    csvfile.close()
    return rh_data


gen = "Words_male"
folder = 1
vid = 1

file_loc = "A:/Project-Sign-Language/ML-Raw-videos/"
pose_dir = file_loc+gen+"/"+str(folder)+"/data/pose_landmarks/"+str(vid)+".csv"
lh_dir = file_loc+gen+"/"+str(folder)+"/data/left_hand_landmarks/"+str(vid)+".csv"
rh_dir = file_loc+gen+"/"+str(folder)+"/data/right_hand_landmarks/"+str(vid)+".csv"


pose_data = dat_pose_extractor(pose_dir) 
lh_data = dat_lh_extractor(lh_dir)
rh_data = dat_rh_extractor(rh_dir)
con_data = np.zeros((90, 258))

for v in range(len(con_data)):
    con = np.array(np.concatenate((pose_data[v], lh_data[v], rh_data[v])))
    con_data[v] = con
    

print(pose_data.ndim)
print(con_data.shape)
print(pose_data.shape)
print(lh_data.shape)
print(rh_data.shape)

