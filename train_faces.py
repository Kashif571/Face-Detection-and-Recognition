import cv2
import numpy as np
import os

#Self Entering the People Name
#p =["Johnny Depp","Leonardo DiCaprio","Samuel Jackson", "Tony Stark", "Hassan Zamir"]

#Getting People Names from the Test Data Folder
people = []
for i in os.listdir(r'E:/Kashif Iftikhar/University/FYP/Code/Python/PR New/Test Data'):
    people.append(i)

#Test Data Folder Path
dir= r"E:/Kashif Iftikhar/University/FYP/Code/Python/PR New/Test Data"

features=[]
labels=[]

def trainer_function():
    for person in people:
        #Joining Main Folder to Persons Folder one by one
        personfolder =os.path.join(dir,person)

        #Getting Index Number
        personlabel =people.index(person)

        for image in os.listdir(personfolder):
            #Joining Person folder with Images in it one by one.
            image_path = os.path.join(personfolder, image)
            
            #Reading Image
            image_array = cv2.imread(image_path)
            
            #Coverting Image to Gray 
            gray_Images = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            #Including Haar Cascade 
            haar_cascade=cv2.CascadeClassifier("haar_faces.xml")

            #Detecting Faces
            face_rect=haar_cascade.detectMultiScale(gray_Images, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in face_rect:
                #Extracting Features
                face_roi = gray_Images[y:y+h, x:x+w]
                
                #Appending Features to Features List
                features.append(face_roi)

                #Appending Person Index Number to Labels List
                labels.append(personlabel)


#Executing Function
trainer_function()

print(f'Total Features = {len(features)}')
print(f'Total Labels = {len(labels)}')

#Converting Features and Labels List to Numpy Arrays
features = np.array(features, dtype='object')
labels=np.array(labels)

#Instantiate the Face Recognizer
face_recognizer =  cv2.face.LBPHFaceRecognizer_create()

#Training the face Recognizer
face_recognizer.train(features, labels)

#Saving Features and Labels Array in Separate File
np.save('features.npy', features)
np.save('labels.npy', labels)

#Saving Trained Faces
face_recognizer.save('trained_faces.yml')


print("Training Successful!!!")


# print(people)
# print(f"Features: {len(features)}")
# print(f"Labels: {len(labels)}")

# print("Training Done!!!")

# video = cv2.VideoCapture('videos/sample video.mp4')

# while True:
#     success, originalframe=video.read()

#     grayframe=cv2.cvtColor(originalframe, cv2.COLOR_BGR2GRAY)

#     # cv2.imshow("Gray Video", grayframe)

#     haar_cascade=cv2.CascadeClassifier('haar_face.xml')
#     face_rect=haar_cascade.detectMultiScale(grayframe, scaleFactor=1.1, minNeighbors=3)

#     for (x,y,w,h) in face_rect:
#         cv2.rectangle(originalframe, (x,y), (x+w, y+h), (0,255,0), thickness=2 )


#     cv2.imshow("Detected Video", originalframe)

    

#     if cv2.waitKey(20) and 0xFF == ord('d'):
#         break


# cv2.release()
# cv2.destroyAllWindows()