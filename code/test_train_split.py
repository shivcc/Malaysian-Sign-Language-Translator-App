#this code will generate the test and train variables

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


def extractor_combiner(pose_dir, lh_dir, rh_dir):
    con_data = np.zeros((90, 258))
    pose_data = dat_pose_extractor(pose_dir) 
    lh_data = dat_lh_extractor(lh_dir)
    rh_data = dat_rh_extractor(rh_dir)
    for v in range(len(pose_data)):
        con = np.array(np.concatenate((pose_data[v], lh_data[v], rh_data[v])))
        con_data[v] = con
    return con_data
        
        
    
    
    

#problem: csv files selection
fold_num = range(1,99)
x_train_num = range(1,4)
x_test_num = 4
gender = ["Words_male", "Words_female"]

x_train = []
x_test = []
y_train = []
y_test = []
gestures = np.array(['saya','beliau','yang','beliau','untuk','pada','dengan','mereka','di','satu','mempunyai','ini','daripada','panas','perkataan','tetapi','apa','beberapa','anda','atau','kepada','dan','dalam','kami','boleh','keluar','lain','yang','masa','jika','akan','bagaimana','setiap','memberitahu','tiga','mahu','baik','bermain','kecil','akhir','meletakkan','rumah','membaca','tangan','besar','menambah','walaupun','tanah','di sini','mesti','tinggi','mengikuti','tindakan','mengapa','meminta','lelaki','perubahan','cahaya','tutup','perlu','rumah','gambar','cuba','kami','lagi','haiwan','titik','ibu','dunia','berhampiran','diri','bumi','bapa','apa-apa','baru','kerja','mengambil','mendapatkan','tempat','hidup','di mana','selepas','belakang','tahun','persembahan','memberi','kami','di bawah','nama','sangat','ayat','bagus','berfikir','tolong','talian','pusing','punca','banyak'])
gesture_map = {label:num for num, label in enumerate(gestures)}


# code x_train lets take vids 1-3 as training data
for folder in fold_num:
     for vid in x_train_num:
         file_loc = "A:/Project-Sign-Language/ML-Raw-videos/"
         for gen in gender:
             pose_dir = file_loc+gen+"/"+str(folder)+"/data/pose_landmarks/"+str(vid)+".csv"
             lh_dir = file_loc+gen+"/"+str(folder)+"/data/left_hand_landmarks/"+str(vid)+".csv"
             rh_dir = file_loc+gen+"/"+str(folder)+"/data/right_hand_landmarks/"+str(vid)+".csv"
             vid_data = extractor_combiner(pose_dir, lh_dir, rh_dir)
             x_train.append(vid_data)
        
#code for x_test  
for folder in fold_num:
     file_loc = "A:/Project-Sign-Language/ML-Raw-videos/"
     for gen in gender:
         pose_dir = file_loc+gen+"/"+str(folder)+"/data/pose_landmarks/"+str(x_test_num)+".csv"
         lh_dir = file_loc+gen+"/"+str(folder)+"/data/left_hand_landmarks/"+str(x_test_num)+".csv"
         rh_dir = file_loc+gen+"/"+str(folder)+"/data/right_hand_landmarks/"+str(x_test_num)+".csv"
         vid_data = extractor_combiner(pose_dir, lh_dir, rh_dir)
         x_test.append(vid_data)
            
#code for y_train          
for ges in gestures:
    for vid in x_train_num:
         for gen in gender:
             y_train.append(gesture_map[ges])
             

#code for y_test      
for ges in gestures:
    for gen in gender:
        y_test.append(gesture_map[ges])
             
y_test = to_categorical(y_test).astype(int)
y_train = to_categorical(y_train).astype(int)
            


#save em at .npy files
path_to_export = "A:/Project-Sign-Language/models/ver6/"
np.save(path_to_export+"x_train.npy", x_train)
np.save(path_to_export+"x_test.npy", x_test)
np.save(path_to_export+"y_train.npy", y_train)
np.save(path_to_export+"y_test.npy", y_test )

print(np.array(np.array(x_train).shape))
print(np.array(np.array(x_test).shape))
print(np.array(np.array(y_train).shape))  
print(np.array(np.array(y_test).shape))





        


