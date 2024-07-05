import cv2
import csv
import mediapipe as mp
import os



#global variables
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
fold_num = range(1,98)
vid_num = range(1,5)
new_vid = range(5,9)  #updated for flipped data


#mediapipe function
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
    
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


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")

folderss = ["Out_source", "Words_male", "Words_female"]

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    for folder in folderss:
        for x in fold_num:
            base_path = f"A:/Project-Sign-Language/ML-Raw-videos/{folder}/"
            path1 = f"{base_path}{x}/data1"
            create_directory_if_not_exists(path1)

            path2 = f"{path1}/right_hand_landmarks"
            create_directory_if_not_exists(path2)
            print(f"folder = {x} in {folder}")

            for y, r in zip(vid_num, new_vid):
                MP4_name = f"{base_path}{x}/{y}.mp4"
                file_locasi = f"{path2}/{r}.csv"

                with open(file_locasi, 'w', newline='') as file:
                    # Write the header for the file
                    header = ["x", "y", "z"]
                    writer = csv.DictWriter(file, fieldnames=header)
                    writer.writeheader()

                    # Cap is opened
                    cap = cv2.VideoCapture(MP4_name)
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            print("Failed to read the frame.")
                            break

                        new_frame = frame[0:720, 320:960]
                        flip_frame = cv2.flip(new_frame, 1)

                        # Make detections
                        image, results = mediapipe_detection(flip_frame, holistic)

                        if hasattr(results.right_hand_landmarks, "landmark"):
                            for data_point in results.right_hand_landmarks.landmark:
                                row = {"x": float(data_point.x), 
                                       "y": float(data_point.y), 
                                       "z": float(data_point.z)}
                                writer.writerow(row)
                            writer.writerow({"x": "x", "y": "x", "z": "x"})
                        else:
                            writer.writerow({"x": "E", "y": "E", "z": "E"})



#main loops
#with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
#     for x in fold_num:
#         s = "A:/Project-Sign-Language/ML-Raw-videos/Out_source/"
#         path1 = s + str(x)+"/data1"
#         create_directory_if_not_exists(path1) 
#    
#         path2 = s + str(x)+"/data1/right_hand_landmarks"
#         create_directory_if_not_exists(path2)
#         print("folder = " +str(x))
#         
#         for y, r in zip(vid_num, new_vid):              
#             MP4_name = s + str(x)+"/"+ str(y)+ ".mp4"
#             file_locasi = path2+"/"+ str(y)+".csv"
#             
#             with open(file_locasi, 'w', newline='') as file:
#                 #writhe the header for the file
#                 header = ["x", "y", "z"]                         
#                 writer = csv.DictWriter(file, fieldnames=header)
#                 writer.writeheader()
#                 
#                 
#                 #print(MP4_name)
#                 #cap is opened
#                 cap = cv2.VideoCapture(MP4_name)
#                 while cap.isOpened():
#                     # Set mediapipe model 
#
#                    
#
#                     # Read feed
#                     ret, frame = cap.read()
#                    
#                     if not ret or frame is None:
#                        print("Failed to read the frame.")
#                        break
#                     
#                     new_frame = frame[0:720 , 320:960]
#                     # Flip the frame
#                     flip_frame = cv2.flip(new_frame, 1)
#                     
#
#
#
#                     # Make detections
#                     image, results = mediapipe_detection(new_frame, holistic)
#                     i = 0  
#                     # Draw landmarks
#                     #draw_styled_landmarks(image, results)
#
#
#
#                     # Show to screen
#                     #cv2.imshow('OpenCV Feed', image)
#                     
#                     if hasattr(results.right_hand_landmarks, "landmark") == True:
#                         for data_point in results.right_hand_landmarks.landmark:
#                             x_point = float(data_point.x)
#                             y_point = float(data_point.y)
#                             z_point = float(data_point.z)
#                            
#                             #print(i, ' x is', x_point, 'y is', y_point, 'z is', z_point,'visibility is', v_point)
#                             row1 = {"x":x_point, "y":y_point, "z":z_point}
#                             writer.writerow(row1)
#                             i+=1
#                         rowx = {"x":"x", "y":"x", "z":"x"}
#                         writer.writerow(rowx)
#                     else:
#                         rowE = {"x":"E", "y":"E", "z":"E"}
#                         writer.writerow(rowE)
                         
         
             
         
      
          
          
     