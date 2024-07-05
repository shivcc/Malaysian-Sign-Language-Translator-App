from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import numpy as np

#general variables
gestures = np.array(['saya','beliau','yang','beliau','untuk','pada','dengan','mereka','di','satu','mempunyai','ini','daripada','panas','perkataan','tetapi','apa','beberapa','anda','atau','kepada','dan','dalam','kami','boleh','keluar','lain','yang','masa','jika','akan','bagaimana','setiap','memberitahu','tiga','mahu','baik','bermain','kecil','akhir','meletakkan','rumah','membaca','tangan','besar','menambah','walaupun','tanah','di sini','mesti','tinggi','mengikuti','tindakan','mengapa','meminta','lelaki','perubahan','cahaya','tutup','perlu','rumah','gambar','cuba','kami','lagi','haiwan','titik','ibu','dunia','berhampiran','diri','bumi','bapa','apa-apa','baru','kerja','mengambil','mendapatkan','tempat','hidup','di mana','selepas','belakang','tahun','persembahan','memberi','kami','di bawah','nama','sangat','ayat','bagus','berfikir','tolong','talian','pusing','punca','banyak'])
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)



#split variables
#path_to_export = "A:/Project-Sign-Language/code/exported_data/"
#sequences = np.load(path_to_export+"sequences.npy")
#labels = np.load(path_to_export+"labels.npy")
#
#
##split the data
#x = np.array(sequences)
#y = to_categorical(labels).astype(int)
#
##print(x)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
#

#use loaded one instead if generating
path_to_export = "A:/Project-Sign-Language/models/ver6/"


x_train = np.load(path_to_export+"x_train.npy")
x_test = np.load(path_to_export+"x_test.npy")
y_train = np.load(path_to_export+"y_train.npy")
y_test = np.load(path_to_export+"y_test.npy")




#model for training
model = Sequential()
model.add(LSTM(5*2, return_sequences=True, activation='relu', input_shape=(90,258)))
model.add(LSTM(5*5, return_sequences=True, activation='relu'))
model.add(LSTM(5*9, return_sequences=True, activation='relu'))
model.add(LSTM(5*2, return_sequences=True, activation='relu'))
model.add(Dense(5*4, activation='relu'))
model.add(Dense(5*4, activation='relu'))
model.add(LSTM(5*3, return_sequences=True, activation='relu'))
model.add(LSTM(5*3, return_sequences=True, activation='relu'))
model.add(LSTM(5*3, return_sequences=False, activation='relu'))
model.add(Dense(5*1, activation='relu'))
model.add(Dense(5*1, activation='relu'))
model.add(Dense(gestures.shape[0], activation='softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])

model.summary()

model.save('A:/Project-Sign-Language/models/ver6/Sign.keras')

