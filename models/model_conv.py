import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
import numpy as np


ver = 29


#model = Sequential()
#model.add(Conv1D(filters=258, kernel_size=5, input_shape=(95, 258),activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
#model.add(MaxPooling1D(pool_size=2))
#model.add(BatchNormalization())
#
#model.add(Conv1D(filters=128, kernel_size=5,activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
#model.add(MaxPooling1D(pool_size=2))
#model.add(BatchNormalization())
#
#
#model.add(LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))
#model.add(Dropout(0.3))
#model.add(BatchNormalization())
#
#model.add(LSTM(64, kernel_regularizer=regularizers.l2(0.001)))
#model.add(Dropout(0.3))
#model.add(BatchNormalization())
#
#
#model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
#model.add(Dense(73, activation='softmax'))
#
#model.load_weights('A:/Project-Sign-Language/models/ver'+ str(ver)+'/Sign.keras')
model = load_model('A:/Project-Sign-Language/models/ver'+ str(ver)+'/Sign.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()
open("A:/Project-Sign-Language/models/ver"+ str(ver)+"/Sign.tflite", "wb").write(tflite_model)

#interpreter = tf.lite.Interpreter(model_path = "Sign_model.tflite")
#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()
#print("Input Shape:", input_details[0]['shape'])
#print("Input Type:", input_details[0]['dtype'])
#print("Output Shape:", output_details[0]['shape'])
#print("Output Type:", output_details[0]['dtype'])