import numpy as np



ver = 26

path_to_export = "A:/Project-Sign-Language/models/ver"+ str(ver)+"/"


x_train = np.load(path_to_export+"x_train.npy")
x_test = np.load(path_to_export+"x_test.npy")
y_train = np.load(path_to_export+"y_train.npy")
y_test = np.load(path_to_export+"y_test.npy")



#def augment_data(x, y, factor=100):
#    augmented_x = []
#    augmented_y = []
#    for i in range(len(x)):
#        for _ in range(factor):
#            # Apply data augmentation techniques here
#            augmented_sample = shift_pos(x[i], 2)
#            augmented_x.append(augmented_sample)
#            augmented_y.append(y[i])  # Make sure to keep corresponding labels
#    return np.array(augmented_x), np.array(augmented_y)

#pose_dat = np.array_split(x_train, 33, axis=2)
pose_dat = x_train[:,:,:33*4]
lh = x_train[:,:,(33*4):(33*4+21*3)]
rh = x_train[:,:,(33*4+21*3):(33*4+21*3+21*3)]
print(x_train.shape)
print(pose_dat.shape)
print(lh.shape)
print(rh.shape)

#def shift_pos(sample, shift_factor):
    
    
    
    #shifting = 
    #shifted_sample = sample* (1+shifting)  