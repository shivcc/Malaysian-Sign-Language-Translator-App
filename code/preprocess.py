#this code will convert the csv files into flatened np array for the x input of the model

import csv
import numpy as np




#This code will be used in the training model code for reading data
# standardize video size ie 3 sec 90frames

frame_size = 95

def dat_pose_extractor(csv_file):
    
    #opens the csv data
    csvfile = open(csv_file , newline='')
    spamreader = csv.reader(csvfile)
    pose_data = np.array(np.zeros((frame_size,4*33)))
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
    lh_data = np.array(np.zeros((frame_size,3*21)))
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
    rh_data = np.array(np.zeros((frame_size,3*21)))
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
    #con_data = np.zeros((frame_size, 258))
    pose_data = dat_pose_extractor(pose_dir) 
    lh_data = dat_lh_extractor(lh_dir)
    rh_data = dat_rh_extractor(rh_dir)
   
    con_data = np.array(np.concatenate((pose_data, lh_data, rh_data), axis=1))
    #con_data[v] = con
    return con_data
        

    
    
    

#problem: csv files selection
fold_num = [2,3,6,7,8,10,11,12,13,14,15,16,17,22,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,43,44,45,47,48,49,51,54,56,57,58,59,60,61,62,64,65,68,69,70,71,72,73,74,76,77,78,80,81,82,84,85,86,88,89,91,92,93,94,95,96]
#fold_num = [3, 6, 8, 11, 12, 14, 16, 17, 28, 31, 33, 36, 38, 43, 56, 58, 61, 68, 69, 70, 71, 73, 77, 80, 88, 89, 91, 92, 94, 96]
vid_num = range(1,9)
gender = ["Words_male", "Words_female", "Out_source"]
sequences, labels = [], []
gestures = np.array(['saya','beliau','yang','beliau','untuk','pada','dengan','mereka','di','satu','mempunyai','ini','daripada','panas','perkataan','tetapi','apa','beberapa','anda','atau','kepada','dan','dalam','kami','boleh','keluar','lain','yang','masa','jika','akan','bagaimana','setiap','memberitahu','tiga','mahu','baik','bermain','kecil','akhir','meletakkan','rumah','membaca','tangan','besar','menambah','walaupun','tanah','di sini','mesti','tinggi','mengikuti','tindakan','mengapa','meminta','lelaki','perubahan','cahaya','tutup','perlu','rumah','gambar','cuba','kami','lagi','haiwan','titik','ibu','dunia','berhampiran','diri','bumi','bapa','apa-apa','baru','kerja','mengambil','mendapatkan','tempat','hidup','di mana','selepas','belakang','tahun','persembahan','memberi','kami','di bawah','nama','sangat','ayat','berfikir','tolong','talian','pusing','punca','banyak'])
gestures_new = [gestures[i-1] for i in fold_num]

gesture_map = {label:num for num, label in enumerate(gestures_new)}
#print(gestures_new)


for folder in fold_num:
     for vid in vid_num:
         file_loc = "A:/Project-Sign-Language/ML-Raw-videos/"
         for gen in gender:
             pose_dir = file_loc+gen+"/"+str(folder)+"/data1/pose_landmarks/"+str(vid)+".csv"
             lh_dir = file_loc+gen+"/"+str(folder)+"/data1/left_hand_landmarks/"+str(vid)+".csv"
             rh_dir = file_loc+gen+"/"+str(folder)+"/data1/right_hand_landmarks/"+str(vid)+".csv"
             print ("folder = " + str(folder)+ " vid = " +str(vid) +" gen = "+str(gen))
             vid_data = extractor_combiner(pose_dir, lh_dir, rh_dir)
             sequences.append(vid_data)
         
        
              
            
for ges in gestures_new:
    for vid in vid_num:
         for gen in gender:
             labels.append(gesture_map[ges])
            


#save em at .npy files
path_to_export = "A:/Project-Sign-Language/exported_data/10/"
np.save(path_to_export+"sequences.npy", sequences)
np.save(path_to_export+"labels.npy", labels)

print(np.array(sequences).shape)
print(np.array(labels).shape)
    

        




        


