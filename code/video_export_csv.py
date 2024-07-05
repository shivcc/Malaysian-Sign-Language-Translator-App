import cv2
import csv
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
 
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
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
    

#cap = cv2.VideoCapture('ML-Raw-videos/Words female/32/1.MP4')
#cap = cv2.VideoCapture('ML-Raw-videos/Words female/79/1.MP4')
file_num = 60
cap = cv2.VideoCapture('A:/Project-Sign-Language/ML-Raw-videos/Out_source/'+str(file_num)+'/3.MP4')
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    header = ["x", "y", "z", "v"]
    file_loc = "A:/Project-Sign-Language/ML-Raw-videos/Out_source/"+str(file_num)+"/data/pose_landmarks/3.csv"
    with open(file_loc, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        while cap.isOpened():
            

         # Read feed
         ret, frame = cap.read()
         #frame = cv2.resize(frame, (960, 540))
         
         frame1 = cv2.flip(frame, 1)
 
         # Make detections
         image, results = mediapipe_detection(frame, holistic)
         i = 0  
         
         if hasattr(results.pose_landmarks, "landmark") == True:     
             for data_point in results.pose_landmarks.landmark:
                 x_point = float(data_point.x)
                 y_point = float(data_point.y)
                 z_point = float(data_point.z)
                 v_point = float(data_point.visibility)
                 print(i, ' x is', x_point, 'y is', y_point, 'z is', z_point,
                  'visibility is', v_point)
                 row1 = {"x":x_point, "y":y_point, "z":z_point, "v":v_point}
                 writer.writerow(row1)
                 i+=1
             rowx = {"x":"x", "y":"x", "z":"x", "v":"x"}
             writer.writerow(rowx)
                
             
             
             # Draw landmarks
             #draw_styled_landmarks(image, results)
             
             
     
             # Show to screen
             #cv2.imshow('OpenCV Feed', image)
             
             
     
             # Break gracefully
             if cv2.waitKey(10) & 0xFF == ord('q'):
                 break
         
        
    cap.release()
    cv2.destroyAllWindows()    

    