# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:54:57 2021
@author: Cao Van Vinh
NHận diện khuôn mặt với OpenCV

"""

import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

pathImg="dataSet"
imagePaths=[os.path.join("dataSet",f) for f in os.listdir("dataSet")]

faces=[]
IDs=[]

print("Training model...")
countImg = 0
classImg = 1
arrImg = []
for imagePath in imagePaths:
     faceImg=Image.open(imagePath).convert('L');
     faceNp=np.array(faceImg,'uint8')
     #split to get ID of the image
     ID=int(os.path.split(imagePath)[-1].split('.')[1])
     if(classImg==ID):
         countImg = countImg+1
     else:
         arrImg.append(countImg)
         countImg=1
         classImg=ID

     faces.append(faceNp)
     IDs.append(ID)
     cv2.imshow("traning",faceNp)
     cv2.waitKey(10)

arrImg.append(countImg)
for i in range(1,len(arrImg)+1):
    print("Class "+str(i)+": "+str(arrImg[i-1])+" image trained")

recognizer.train(faces,np.array(IDs))
if not os.path.exists("recognizer"):
    os.makedirs("recognizer")
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()
print("Finished training!")

"""
def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        #split to get ID of the image
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print(ID)
        IDs.append(ID)
        cv2.imshow("traning",faceNp)
        cv2.waitKey(10)
    return IDs, faces

Ids,faces=getImagesAndLabels(path)
#trainning
recognizer.train(faces,np.array(Ids))
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()
"""


