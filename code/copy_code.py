import shutil


fold_num = range(1,98)
vid_num = range(1,5)
new_vid = range(5,9)


for x in fold_num:
         s = "A:/Project-Sign-Language/ML-Raw-videos/Out_source/"
         path1 = s + str(x)+"/data"
         #os.mkdir(path1) 
    
         path2 = s + str(x)+"/data/left_hand_landmarks"
         #os.mkdir(path2)
         print("folder = " +str(x))
         
         for y, r in zip(vid_num, new_vid):
             source_file = 'A:/Project-Sign-Language/ML-Raw-videos/Out_source/'+str(x)+'/data/right_hand_landmarks/'+str(y)+'.csv'
             destination_file = 'A:/Project-Sign-Language/ML-Raw-videos/Out_source/'+str(x)+'/data/left_hand_landmarks/'+str(r)+'.csv'     
             # Copy and rename the file
             shutil.copy(source_file, destination_file)         
             
