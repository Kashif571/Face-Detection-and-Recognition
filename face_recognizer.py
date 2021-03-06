import cv2
import numpy as np
import os

haar_cascade=cv2.CascadeClassifier('haar_faces.xml')

#Getting People Names from the Test Data Folder
people =[]
for i in os.listdir(r'E:/Kashif Iftikhar/University/FYP/Code/Python/PR New/Test Data/'):
    people.append(i)

#Instantiate the Face Recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Reading Trained Face
face_recognizer.read('trained_faces.yml')

#Reading Image
img=cv2.imread(r'E:/Kashif Iftikhar/University/FYP/Code/Python/PR New/Validation Data/Hassan Zamir/4.jpg')
#Adding Blur
# blur_Image = cv2.medianBlur(img, 9)

#Converting to Gray Scale
gray_Image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#Detecting Faces
face_rect=haar_cascade.detectMultiScale(gray_Image, scaleFactor=1.1, minNeighbors=4)

for (x,y,w,h) in face_rect:
    #Getting Face Area
    face_roi=gray_Image[y:y+h, x:x+w]

    #Getting Match Value
    label, confidence = face_recognizer.predict(face_roi)
    print(label, confidence)

    #Displaying Text and Box
    cv2.putText(img, str(people[label]), (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    # if confidence > 0:
    #     cv2.putText(img, str(people[label]), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)

    #     cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    # else:
    #     cv2.putText(img, str('Unknown'), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)

    #     cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)


#Showng Image with Detection
cv2.imshow("Detected Face",img)
cv2.waitKey(0)

