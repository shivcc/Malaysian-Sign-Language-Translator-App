from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras import regularizers
import numpy as np

ver = 29


#split variables
path_to_export = "A:/Project-Sign-Language/code/exported_data/"
#sequences = np.load(path_to_export+"sequences.npy")
#labels = np.load(path_to_export+"labels.npy")
gestures = np.array(['saya','beliau','yang','beliau','untuk','pada','dengan','mereka','di','satu','mempunyai','ini','daripada','panas','perkataan','tetapi','apa','beberapa','anda','atau','kepada','dan','dalam','kami','boleh','keluar','lain','yang','masa','jika','akan','bagaimana','setiap','memberitahu','tiga','mahu','baik','bermain','kecil','akhir','meletakkan','rumah','membaca','tangan','besar','menambah','walaupun','tanah','di sini','mesti','tinggi','mengikuti','tindakan','mengapa','meminta','lelaki','perubahan','cahaya','tutup','perlu','rumah','gambar','cuba','kami','lagi','haiwan','titik','ibu','dunia','berhampiran','diri','bumi','bapa','apa-apa','baru','kerja','mengambil','mendapatkan','tempat','hidup','di mana','selepas','belakang','tahun','persembahan','memberi','kami','di bawah','nama','sangat','ayat','berfikir','tolong','talian','pusing','punca','banyak'])


#split the data
#x = np.array(sequences)
#y = to_categorical(labels).astype(int)

x_train = np.load("A:/Project-Sign-Language/models/ver"+ str(ver)+"/x_train.npy")
x_test =  np.load("A:/Project-Sign-Language/models/ver"+ str(ver)+"/x_test.npy")
y_train = np.load("A:/Project-Sign-Language/models/ver"+ str(ver)+"/y_train.npy")
y_test =  np.load("A:/Project-Sign-Language/models/ver"+ str(ver)+"/y_test.npy")

print(np.array(x_train).shape)
print(np.array(x_test ).shape)
print(np.array(y_train).shape)
print(np.array(y_test ).shape)


model = Sequential()

# Define the model
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
model.add(Dense(72, activation='softmax'))


    


model.load_weights('A:/Project-Sign-Language/models/ver'+ str(ver)+'/Sign.keras')
res = model.predict(x_test)

yhat = model.predict(x_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

#print(multilabel_confusion_matrix(ytrue, yhat))

print(accuracy_score(ytrue, yhat))
print(multilabel_confusion_matrix(ytrue, yhat))