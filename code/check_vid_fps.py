import cv2
import numpy as np

fold_num = range(1,98)
vid_num = range(1,5)

s = "A:/Project-Sign-Language/ML-Raw-videos/Out_source/"


maximf = []
for x in fold_num:
    for y in vid_num:
        vid = s + str(x)+"/"+ str(y)+ ".mp4"
        cap = cv2.VideoCapture(vid)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:print(vid)
        totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        durationInSeconds = totalNoFrames//fps
        maximf.append(durationInSeconds)
        print(np.argmax(maximf))
        print(vid)
        
