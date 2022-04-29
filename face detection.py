import numpy as np #pip install numpy in cmd
import cv2 # pip install numpy in cmd

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #face model
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') #eye model
cap = cv2.VideoCapture(0) #switch on camera in video mode
while True:
    boool, img = cap.read() #segmenting video into frame by frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert RGB to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    #detection and mapping of face
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        #detection and mapping of eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'): #stoping the loop
        break
cap.release() #switch off camera
cv2.destroyAllWindows() #closing all windows created using cv2 library
