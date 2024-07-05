import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#general variables
gestures = np.array(['saya','beliau','yang','beliau','untuk','pada','dengan','mereka','di','satu','mempunyai','ini','daripada','panas','perkataan','tetapi','apa','beberapa','anda','atau','kepada','dan','dalam','kami','boleh','keluar','lain','yang','masa','jika','akan','bagaimana','setiap','memberitahu','tiga','mahu','baik','bermain','kecil','akhir','meletakkan','rumah','membaca','tangan','besar','menambah','walaupun','tanah','di sini','mesti','tinggi','mengikuti','tindakan','mengapa','meminta','lelaki','perubahan','cahaya','tutup','perlu','rumah','gambar','cuba','kami','lagi','haiwan','titik','ibu','dunia','berhampiran','diri','bumi','bapa','apa-apa','baru','kerja','mengambil','mendapatkan','tempat','hidup','di mana','selepas','belakang','tahun','persembahan','baik','memberi','kami','di bawah','nama','sangat','ayat','bagus','berfikir','tolong','talian','pusing','punca','banyak'])




#split variables
path_to_export = "A:/Project-Sign-Language/code/exported_data/"
sequences = np.load(path_to_export+"sequences.npy")
labels = np.load(path_to_export+"labels.npy")


#split the data
x = np.array(sequences)
y = to_categorical(labels).astype(int)


print(x.shape)
print(y.shape)


#generate and observe shape
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#load generated and observe shape
x_train = np.load("A:/Project-Sign-Language/code/exported_data/x_train.npy")
x_test = np.load("A:/Project-Sign-Language/code/exported_data/x_test.npy")
y_train = np.load("A:/Project-Sign-Language/code/exported_data/y_train.npy")
y_test = np.load("A:/Project-Sign-Language/code/exported_data/y_test.npy")



print("1 x: "+ str(x.shape))
print("2 y: "+ str(y.shape))
print("3 x_train: "+ str(np.array(x_train).shape))
print("4 x_test: "+ str(np.array(x_test ).shape))
print("5 y_train: "+ str(np.array(y_train ).shape))
print("6 y_test: "+ str(np.array(y_test).shape))