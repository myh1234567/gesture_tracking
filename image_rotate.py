from PIL import Image
import os
import cv2
i = 2000
for filename in os.listdir(r"C:\Users\group\Desktop\Hand-Gesture-Recognizer\dataset\training_set\thumbs_up"):
    print(filename)
    im = Image.open(r"C:\Users\group\Desktop\Hand-Gesture-Recognizer\dataset\training_set\thumbs_up" + "\\" + filename )
    im_rotate = im.rotate(180)
    im_rotate.save(r"C:\Users\group\Desktop\Hand-Gesture-Recognizer\dataset\training_set\thumbs_up"+ "\\" + str(i) + '.jpg')
    i+=1
