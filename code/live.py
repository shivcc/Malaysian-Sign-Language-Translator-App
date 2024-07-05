import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras import regularizers
import numpy as np
import mediapipe as mp

#general variables
#fold_num = [2,3,6,7,8,10,11,12,13,14,15,16,17,22,25,26,27,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,43,44,45,47,48,49,51,54,56,57,58,59,60,61,62,64,65,68,69,70,71,72,73,74,76,77,78,80,81,82,84,85,86,88,89,91,92,93,94,95,96]
fold_num = [3, 6, 8, 11, 12, 14, 16, 17, 28, 31, 33, 36, 38, 43, 56, 58, 61, 68, 69, 70, 71, 73, 77, 80, 88, 89, 91, 92, 94, 96]
gestures1 = np.array(['saya','beliau','yang','beliau','untuk','pada','dengan','mereka','di','satu','mempunyai','ini','daripada','panas','perkataan','tetapi','apa','beberapa','anda','atau','kepada','dan','dalam','kami','boleh','keluar','lain','yang','masa','jika','akan','bagaimana','setiap','memberitahu','tiga','mahu','baik','bermain','kecil','akhir','meletakkan','rumah','membaca','tangan','besar','menambah','walaupun','tanah','di sini','mesti','tinggi','mengikuti','tindakan','mengapa','meminta','lelaki','perubahan','cahaya','tutup','perlu','rumah','gambar','cuba','kami','lagi','haiwan','titik','ibu','dunia','berhampiran','diri','bumi','bapa','apa-apa','baru','kerja','mengambil','mendapatkan','tempat','hidup','di mana','selepas','belakang','tahun','persembahan','memberi','kami','di bawah','nama','sangat','ayat','berfikir','tolong','talian','pusing','punca','banyak'])
gestures = [gestures1[i-1] for i in fold_num]
ver = 28
frame_size = 95


#global variables
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


#mediapipe function
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
    
def draw_styled_landmarks(image, results):
    ## Draw face connections
    #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
    #                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    #                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                         ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 




#def extract_keypoints(results):
#    return np.array(np.concatenate((extract_pose(results), extract_lh(results), extract_rh(results))))
#
#def extract_pose (results):
#    if results.pose_landmarks:
#        for i in results.pose_landmarks.landmark:
#            pose_dat = [i.x, i.y, i.z, i.visibility]   
#    else:
#        pose_dat = np.zeros(33*4)    
#    return pose_dat
#                       
#def extract_lh (results):
#    if results.left_hand_landmarks:
#        for i in results.left_hand_landmarks.landmark:
#            lh_dat = [i.x, i.y, i.z]
#    else:
#        lh_dat = np.zeros(21*3)   
#    return lh_dat            
#        
#def extract_rh (results):
#    if results.right_hand_landmarks:
#        for i in results.right_hand_landmarks.landmark:
#            rh_dat = [i.x, i.y, i.z]
#    else:
#        rh_dat = np.zeros(21*3)
#    return rh_dat  


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

##split variables
#path_to_export = "A:/Project-Sign-Language/code/exported_data/"
#sequences = np.load(path_to_export+"sequences.npy")
#labels = np.load(path_to_export+"labels.npy")
#
#
##split the data
#x = np.array(sequences)
#y = to_categorical(labels).astype(int)

x_train = np.load("A:/Project-Sign-Language/models/ver"+ str(ver)+"/x_train.npy")
x_test =  np.load("A:/Project-Sign-Language/models/ver"+ str(ver)+"/x_train.npy")
y_train = np.load("A:/Project-Sign-Language/models/ver"+ str(ver)+"/y_train.npy")
y_test =  np.load("A:/Project-Sign-Language/models/ver"+ str(ver)+"/y_test.npy")




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
model.add(Dense(30, activation='softmax'))




model.load_weights('A:/Project-Sign-Language/models/ver'+ str(ver)+'/Sign.keras')







colors = [(134, 145, 200),(255, 128, 0),(0, 128, 255),(128, 0, 255),(255, 0, 128),(128, 255, 0),(0, 255, 128),(255, 255, 0),(0, 255, 255),(255, 0, 255),(192, 192, 192),(128, 128, 128),(128, 0, 0),(128, 128, 0),(0, 128, 0),(128, 0, 128),(0, 128, 128),(0, 0, 128),(255, 255, 255),(0, 0, 0),(165, 42, 42),(255, 127, 80),(255, 69, 0),(255, 215, 0),(50, 205, 50),(173, 255, 47),(0, 255, 127),(144, 238, 144),(60, 179, 113),(46, 139, 87),(34, 139, 34),(0, 128, 0),(0, 100, 0),(154, 205, 50),(85, 107, 47),(107, 142, 35),(128, 128, 0),(189, 183, 107),(240, 230, 140),(238, 232, 170),(255, 255, 224),(255, 250, 205),(250, 250, 210),(255, 255, 0),(255, 215, 0),(255, 69, 0),(255, 99, 71),(255, 127, 80),(255, 140, 0),(255, 165, 0),(255, 182, 193),(255, 192, 203),(250, 235, 215),(245, 245, 220),(255, 228, 181),(255, 235, 205),(255, 248, 220),(255, 250, 205),(255, 250, 240),(255, 255, 240),(255, 255, 224),(255, 255, 255),(0, 0, 0),(47, 79, 79),(105, 105, 105),(112, 128, 144),(119, 136, 153),(190, 190, 190),(211, 211, 211),(220, 220, 220),(245, 245, 245),(255, 250, 250),(240, 255, 240),(245, 255, 250),(240, 255, 255),(240, 248, 255),(248, 248, 255),(230, 230, 250),(255, 240, 245),(240, 128, 128),(255, 182, 193),(250, 128, 114),(233, 150, 122),(255, 160, 122),(255, 69, 0),(255, 99, 71),(255, 127, 80),(205, 92, 92),(250, 235, 215),(245, 245, 220),(255, 228, 196),(255, 235, 205),(255, 222, 173),(255, 218, 185),(255, 250, 205),(250, 250, 210),(255, 255, 224),(255, 255, 240),(255, 255, 255)]

def prob_viz(res, gestures, input_frame, colors):
    output_frame = input_frame.copy()
    #kes = np.array(res)
    #for num in range(1,4):
    #    cv2.rectangle(output_frame, (0,60+num*40), (int(kes[num]*100), 90+num*40), colors[num], -1)
    #    cv2.putText(output_frame, gestures[np.argmax(kes[role])], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, gestures[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame



# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.7  # Adjust the threshold based on your needs

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-95:]

        if len(sequence) == 95:
            pred = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(gestures[np.argmax(pred)])
            predictions.append(np.argmax(pred))

            # 3. Viz logic
            if pred[np.argmax(pred)] > threshold:
                if len(sentence) > 0:
                    if gestures[np.argmax(pred)] != sentence[-1]:
                        sentence.append(gestures[np.argmax(pred)])
                else:
                    sentence.append(gestures[np.argmax(pred)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]
            

            # Viz probabilities
            # image = prob_viz(res, gestures, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
