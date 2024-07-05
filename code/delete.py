import os

fold_num = range(1,98)
dat_num = range(1,9)
s = "A:/Project-Sign-Language/ML-Raw-videos/Out_source/"


"""

#delete items inside
for x in fold_num:
    for y in dat_num:
        path_file1 = s + str(x) + "/data/right_hand_landmarks/" + str(y) + ".csv"
        os.remove(path_file1)
        path_file2 = s + str(x) + "/data/left_hand_landmarks/" + str(y) + ".csv"
        os.remove(path_file2)
        path_file3 = s + str(x) + "/data/pose_landmarks/" + str(y) + ".csv"
        os.remove(path_file3)
        
#note that this gives error if the file does not exist        


"""
#for deleting folders: /data

for x in fold_num:
    path_folder = s+str(x)+"/data/right_hand_landmarks"
    os.rmdir(path_folder)
    path_folder1 = s+str(x)+"/data/left_hand_landmarks"
    os.rmdir(path_folder1)
    path_folder2 = s+str(x)+"/data/pose_landmarks"
    os.rmdir(path_folder2)



