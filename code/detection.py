from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
import numpy as np

#general variables
fold_num = [2,3,6,7,8,10,11,12,13,14,15,16,17,22,25,26,27,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,43,44,45,47,48,49,51,54,56,57,58,59,60,61,62,64,65,68,69,70,71,72,73,74,76,77,78,80,81,82,84,85,86,88,89,91,92,93,94,95,96]
gestures = np.array(['saya','beliau','yang','beliau','untuk','pada','dengan','mereka','di','satu','mempunyai','ini','daripada','panas','perkataan','tetapi','apa','beberapa','anda','atau','kepada','dan','dalam','kami','boleh','keluar','lain','yang','masa','jika','akan','bagaimana','setiap','memberitahu','tiga','mahu','baik','bermain','kecil','akhir','meletakkan','rumah','membaca','tangan','besar','menambah','walaupun','tanah','di sini','mesti','tinggi','mengikuti','tindakan','mengapa','meminta','lelaki','perubahan','cahaya','tutup','perlu','rumah','gambar','cuba','kami','lagi','haiwan','titik','ibu','dunia','berhampiran','diri','bumi','bapa','apa-apa','baru','kerja','mengambil','mendapatkan','tempat','hidup','di mana','selepas','belakang','tahun','persembahan','memberi','kami','di bawah','nama','sangat','ayat','berfikir','tolong','talian','pusing','punca','banyak'])
gestures_new = [gestures[i-1] for i in fold_num]
print(gestures_new)
ver = 27


#variables


x_train = np.load("A:/Project-Sign-Language/models/ver"+ str(ver)+"/x_train.npy")
x_test =  np.load("A:/Project-Sign-Language/models/ver"+ str(ver)+"/x_test.npy")
y_train = np.load("A:/Project-Sign-Language/models/ver"+ str(ver)+"/y_train.npy")
y_test =  np.load("A:/Project-Sign-Language/models/ver"+ str(ver)+"/y_test.npy")

print(np.array(x_train).shape)
print(np.array(x_test ).shape)
print(np.array(y_train ).shape)
print(np.array(y_test).shape)


model = Sequential()
model.add(Conv1D(filters=258, kernel_size=5, input_shape=(95, 258),activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv1D(filters=128, kernel_size=5,activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())


model.add(LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(LSTM(64, kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.3))
model.add(BatchNormalization())


model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(73, activation='softmax'))

model.load_weights('A:/Project-Sign-Language/models/ver'+ str(ver)+'/Sign.keras')
res = model.predict(x_test)

    
for role in range(len(x_test)):
    print(str(role)+ ". Word:       " +str(gestures_new[np.argmax(y_test[role])]))
    print(str(role)+ ". Prediction: " +str(gestures_new[np.argmax(res[role])]))
    
        
 
 
    
    




