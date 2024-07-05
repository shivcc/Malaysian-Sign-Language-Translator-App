import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import numpy as np
print(tf.__version__)

ver = 29

path_to_export = "A:/Project-Sign-Language/models/ver"+ str(ver)+"/"


x_train = np.load(path_to_export+"x_train.npy")
x_test = np.load(path_to_export+"x_test.npy")
y_train = np.load(path_to_export+"y_train.npy")
y_test = np.load(path_to_export+"y_test.npy")

# Function for noise, time and scale data augmentation
def augment_data(x, y, factor=2):
    augmented_x = []
    augmented_y = []
    for i in range(len(x)):
        for _ in range(factor):
            # Apply data augmentation techniques here
            augmented_sample = add_noise_and_scale(x[i])
            #augmented_sample = shift_pos(x[i], 2)
            augmented_sample = random_time_crop_and_pad(augmented_sample, min_crop_length=80, max_crop_length=95)
            #augmented_sample = add_time_variation(augmented_sample)  # Implement time variation augmentation function
            augmented_x.append(augmented_sample)
            augmented_y.append(y[i])  # Make sure to keep corresponding labels
    return np.array(augmented_x), np.array(augmented_y)



# Data augmentation functions


# Random time cropping augmentation function with padding
def random_time_crop_and_pad(sample, min_crop_length=80, max_crop_length=95):
    """
    Randomly crops a time series sample to a length between min_crop_length and max_crop_length,
    then pads with zeros to max_crop_length.
    
    :param sample: The input time series sample to be cropped.
    :param min_crop_length: The minimum length to crop to.
    :param max_crop_length: The length to pad to after cropping.
    :return: A cropped and padded time series sample.
    """
    assert min_crop_length <= max_crop_length, "min_crop_length must be less than or equal to max_crop_length"
    
    # Determine the crop length
    crop_length = np.random.randint(min_crop_length, max_crop_length + 1)
    
    # Crop the sample
    if crop_length < sample.shape[0]:
        start = np.random.randint(0, sample.shape[0] - crop_length)
        cropped_sample = sample[start:start + crop_length, :]
    else:
        cropped_sample = sample

    # Pad the sample
    if cropped_sample.shape[0] < max_crop_length:
        padding = np.zeros((max_crop_length - cropped_sample.shape[0], cropped_sample.shape[1]))
        padded_sample = np.vstack((cropped_sample, padding))
    else:
        padded_sample = cropped_sample
    
    return padded_sample


# Time variation and noise and scale augmentation functions
def add_time_variation(sample):
    # Apply time variation techniques
    #augmented_sample = apply_time_warping(sample)
    #augmented_sample = apply_time_shifting(augmented_sample)
    augmented_sample = add_noise_and_scale(sample)
    return augmented_sample

# Time variation augmentation techniques
def apply_time_warping(sample, warp_factor=0.3):     #removing cause of suspition
    return np.roll(sample, int(len(sample) * np.random.uniform(-warp_factor, warp_factor)))

def apply_time_shifting(sample, shift_factor=0.5):
    shift = int(len(sample) * np.random.uniform(-shift_factor, shift_factor))
    return np.roll(sample, shift)

def add_noise_and_scale(sample, noise_factor=0.0001, scale_factor=0.2):
    noise = np.random.normal(0, noise_factor, sample.shape)
    scaled_sample = sample * (1 + scale_factor * np.random.uniform(-1, 1))
    return scaled_sample + noise


def shift_pos(sample, shift_scale):
    shifting = np.random.uniform(-1*shift_scale,1*shift_scale)
    return shifting + sample
    
# Apply data augmentation to your data
x_train_augmented, y_train_augmented = augment_data(x_train, y_train)

# Combine augmented data with original data
x_train_combined = np.concatenate((x_train, x_train_augmented), axis=0)
y_train_combined = np.concatenate((y_train, y_train_augmented), axis=0)



model = Sequential()

# '95' is the number of frames in each video and '258' is the number of features per frame.
#model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'), input_shape=(95, 258)))
#model.add(TimeDistributed(Flatten()))  # Flatten the features from Conv1D for each timestep

# LSTM layers with dropout

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



# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('A:/Project-Sign-Language/models/ver' + str(ver) + '/Sign.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00001)

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(
    x_train_combined, y_train_combined, 
    #x_train, y_train, 
    epochs=2000, 
    batch_size=64, 
    validation_split=0.1, 
    callbacks=[TensorBoard(log_dir='./models/ver' + str(ver) + '/logs'), early_stopping, model_checkpoint, reduce_lr], 
    verbose=1
)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

