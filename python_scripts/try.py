import cv2
import math
videoFile = "data/test.avi"
imagesFolder = "data/images"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
count = 0
while(cap.isOpened()):
    #print("Here")
    frameId = cap.get(1) #current frame number
    print(frameId)
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % 6) == 0):
        filename = imagesFolder + "/image_" +  str(int(frameId)) + ".jpg"
        cv2.imwrite(filename, frame)
cap.release()

