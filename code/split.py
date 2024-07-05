from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np


ver = 29
#gestures = np.array(['saya','beliau','yang','beliau','untuk','pada','dengan','mereka','di','satu','mempunyai','ini','daripada','panas','perkataan','tetapi','apa','beberapa','anda','atau','kepada','dan','dalam','kami','boleh','keluar','lain','yang','masa','jika','akan','bagaimana','setiap','memberitahu','tiga','mahu','baik','bermain','kecil','akhir','meletakkan','rumah','membaca','tangan','besar','menambah','walaupun','tanah','di sini','mesti','tinggi','mengikuti','tindakan','mengapa','meminta','lelaki','perubahan','cahaya','tutup','perlu','rumah','gambar','cuba','kami','lagi','haiwan','titik','ibu','dunia','berhampiran','diri','bumi','bapa','apa-apa','baru','kerja','mengambil','mendapatkan','tempat','hidup','di mana','selepas','belakang','tahun','persembahan','memberi','kami','di bawah','nama','sangat','ayat','berfikir','tolong','talian','pusing','punca','banyak'])
path = "A:/Project-Sign-Language/exported_data/10/"
sequences = np.load(path+"sequences.npy")
labels = np.load(path+"labels.npy")

print(sequences.shape)
print(labels.shape)

#split the data
x = np.array(sequences)
y = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


#save em at .npy files
path_to_export = "A:/Project-Sign-Language/models/ver"+ str(ver)+"/"
np.save(path_to_export+"x_train.npy", x_train)
np.save(path_to_export+"x_test.npy", x_test)
np.save(path_to_export+"y_train.npy", y_train)
np.save(path_to_export+"y_test.npy", y_test )

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



