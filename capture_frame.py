import cv2

# cap=cv2.VideoCapture(0)
# i=500
# while(1):
#     ret ,frame = cap.read()
#     k=cv2.waitKey(1)
#     cropImg = frame[200:400, 300:500]
#     if k==27:
#         break
#     elif k==ord('s'):
#         cv2.imwrite('/Users/yuhaomao/Desktop/lu_dianzan/'+str(i)+'.jpg',cropImg)
#         i+=1
#     # cv2.rectangle(frame, (20,20), (400, 400), (0, 255, 0), 1)
#     cv2.rectangle(frame, (300, 200), (500, 400), (0, 255, 0), 1)
#     # frame = cv2.flip(frame, 1)
#     cv2.imshow("capture", frame)
# cap.release()
# cv2.destroyAllWindows()


from PIL import Image

im = Image.open("/Users/yuhaomao/Desktop/lu_dianzan/500.jpg")
im.show()

im_rotate = im.rotate(90)
im_rotate.show()
