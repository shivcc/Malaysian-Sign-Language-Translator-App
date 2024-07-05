# Import packages
import cv2
import numpy as np
 
img = cv2.imread('A:/Project-Sign-Language/code/pic_sign_2.png')
print(img.shape) # Print image shape
cv2.imshow("original", img)
 
# Cropping an image
cropped_image = img[0:720 , 320:960]
 
# Display cropped image
cv2.imshow("cropped", cropped_image)
 
# Save the cropped image
#cv2.imwrite("Cropped Image.jpg", cropped_image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()