from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2 
import numpy as np

ver = 19

path_to_export = "A:/Project-Sign-Language/models/ver"+ str(ver)+"/"


x_train = np.load(path_to_export+"x_train.npy")
x_test = np.load(path_to_export+"x_test.npy")
y_train = np.load(path_to_export+"y_train.npy")
y_test = np.load(path_to_export+"y_test.npy")

# Function for noise, time and scale data augmentation
def augment_data(x, y, factor=3):
    augmented_x = []
    augmented_y = []
    for i in range(len(x)):
        for _ in range(factor):
            # Apply data augmentation techniques here
            augmented_sample = add_time_variation(x[i])  # Implement time variation augmentation function
            augmented_x.append(augmented_sample)
            augmented_y.append(y[i])  # Make sure to keep corresponding labels
    return np.array(augmented_x), np.array(augmented_y)



# Data augmentation functions


# Time variation and noise and scale augmentation functions
def add_time_variation(sample):
    # Apply time variation techniques
    #augmented_sample = apply_time_warping(sample)
    augmented_sample = apply_time_shifting(sample)
    augmented_sample = add_noise_and_scale(augmented_sample)
    return augmented_sample

# Time variation augmentation techniques
#def apply_time_warping(sample, warp_factor=0.1):     #removing cause of suspition
#    return np.roll(sample, int(len(sample) * np.random.uniform(-warp_factor, warp_factor)))

def apply_time_shifting(sample, shift_factor=0.2):
    shift = int(len(sample) * np.random.uniform(-shift_factor, shift_factor))
    return np.roll(sample, shift)

def add_noise_and_scale(sample, noise_factor=0.001, scale_factor=0.2):
    noise = np.random.normal(0, noise_factor, sample.shape)
    scaled_sample = sample * (1 + scale_factor * np.random.uniform(-1, 1))
    return scaled_sample + noise

# Apply data augmentation to your data
x_train_augmented, y_train_augmented = augment_data(x_train, y_train)

# Combine augmented data with original data
x_train_combined = np.concatenate((x_train, x_train_augmented), axis=0)
y_train_combined = np.concatenate((y_train, y_train_augmented), axis=0)



model = Sequential([
    Input(shape=(90, 258)),
    Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    BatchNormalization(),
    
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    BatchNormalization(),
    
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    BatchNormalization(),
    
    LSTM(64),
    Dropout(0.5),
    BatchNormalization(),
    
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(73, activation='softmax')
])
# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('A:/Project-Sign-Language/models/ver' + str(ver) + '/Sign.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00001)

# Compile the model
optimizer = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(
    x_train_combined, y_train_combined, 
    epochs=2000, 
    batch_size=16, 
    validation_split=0.1, 
    callbacks=[TensorBoard(log_dir='./models/ver' + str(ver) + '/logs'), early_stopping, model_checkpoint, reduce_lr], 
    verbose=1
)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

