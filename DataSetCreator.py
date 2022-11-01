# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:54:57 2021
@author: Cao Van Vinh
NHận diện khuôn mặt với OpenCV

"""

import cv2
import sqlite3
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')

#insert/update data to sqlite
def insertOrUpdate(Id,Name,Age,Gender):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE People SET Name='"+str(Name)+"' WHERE ID='"+str(Id)+"'"
    else:
        cmd="INSERT INTO People(Id,Name,Age,Gender) Values('"+str(Id)+"','"+str(Name)+"','"+str(Age)+"','"+str(Gender)+"')"
    conn.execute(cmd)
    conn.commit()
    conn.close()

id=input('enter your id >>:')
name=input('enter your name >>:')
age=input('enter your age >>:')
gender=input('enter your gender >>:')
insertOrUpdate(id,name,age,gender)

sampleNum=0
print("Getting data...")
while(True):
    #camera read
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        #incrementing sample number
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User."+id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('frame',img)
    #wait for 100 miliseconds
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()
print("Data has been retrieved")
