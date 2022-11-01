# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:54:57 2021
@author: Cao Van Vinh
NHận diện khuôn mặt với OpenCV
"""

import cv2,os
import numpy as np
from PIL import Image
import pickle
import sqlite3
from datetime import date


# Load face recognizer
face_recognizer=cv2.face.LBPHFaceRecognizer_create();
face_recognizer.read("recognizer\\trainningData.yml")
#set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 0.6
fontcolor = (203,23,252)
ID=0

#get data from sqlite by ID
def getProfile(id=-0):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row

    if profile!=None and ID!=id:
        DiemDanh(id,profile)
    conn.close()
    return profile

# diem danh khuon mat
def DiemDanh(id,profile):
    conn=sqlite3.connect("FaceBase.db")
    profile1 = None
    ID=id
    ngayHomNay = str(date.today())
    cmd1 = "SELECT*FROM DiemDanh where ID="+str(id)+" and Ngay='"+ngayHomNay+"'"
    cur1 = conn.execute(cmd1)
    for row1 in cur1:
        profile1 = row1
    if(profile1==None):
        cmd2="INSERT INTO DiemDanh(Id,Name,Ngay) Values('"+str(id)+"','"+str(profile[1])+"','"+ngayHomNay+"')"
        conn.execute(cmd2)
        conn.commit()
        conn.close()
        print("[INFO] "+str(profile)+" đã được điểm danh !!!")

# function to detect face using OpenCV
def predict(test_img):
    # make a copy of the image
    img = test_img.copy()
    # detect face from the image
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("[INFO] loading face detector...")
    # load OpenCV face detector use LBP  more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
    print("[INFO] performing face detection...")
    # detect multiscale images
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=1,minSize=(20,20))  # result is a list of faces

    # if no faces are detected then return original img
    if (len(faces) == 0):
        print("[INFO] Không có khuôn mặt nào được phát hiện")
        return None

    for everyface in faces:
        (x,y,w,h) = everyface
        everygray = gray[y:y+w, x:x+h]
        # predict the image using face recognizer
        label, confidence = face_recognizer.predict(everygray)
        # get name of respective label returned by face recognizer
        profile = getProfile(label)
        print("[INFO] Predict: ",str(profile),"\n[INFO] Accuracy: ", str(label), str(confidence))
        # draw a rectangle around face detected
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # draw name of predicted person
        if(profile!=None):
            #cv2.PutText(cv2.fromarray(img),str(id),(x+y+h),font,(0,0,255),2);
            cv2.putText(img, "Name: " + str(profile[1]), (x,y+h+25), fontface, fontscale, fontcolor ,2)
            cv2.putText(img, "Age: " + str(profile[2]), (x,y+h+45), fontface, fontscale, fontcolor ,2)
            cv2.putText(img, "Gender: " + str(profile[3]), (x,y+h+65), fontface, fontscale, fontcolor ,2)
        else:
            cv2.putText(img, "No information!", (x,y+h+30), fontface, fontscale, fontcolor ,2)
            print("[INFO] label_text is null")
    print("[INFO] ",len(faces)," faces detected")
    return img

if __name__ == '__main__':
    print("* ----- Chọn nguồn để nhận dạng:  ----- *")
    print("   ===   1. Từ ảnh đầu vào   ===")
    print("   ===   2. Từ camera        ===")
    while(1):
        try:
            mode = int(input('[INFO] Chọn mô hình nhận diên(1 hoặc 2) >>: '))
        except ValueError:
            print("[INFO] Not a number, program exit")
            break
        if mode == 1:
            while(1):
                path = "anh/a"+input('[INFO] Nhập tên ảnh: ')+".jpg"
                assert os.path.exists(path), "Không tồn tại file, " + str(path)
                # load test images
                test_img = cv2.imread(path)
                # perform a prediction
                predicted_img = predict(test_img) #moi
                if(predicted_img is None):
                    pass
                else:
                    print("[INFO] Prediction complete")
                    # display images
                    cv2.imshow("Result", predicted_img)
                    cv2.waitKey(0)
                cv2.destroyAllWindows()
                try:
                    con = int(input('[INFO] Bạn có muốn tiếp tục (nhập một số; 0 để kết thúc) >>:'))
                    if con == 0 :
                        break
                except ValueError:
                    print("[INFO] Not a number")
                    break
        elif mode == 2:
            print("[INFO] loading face detector...")
            cam=cv2.VideoCapture(0);
            faceDetect=cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml');
            eye_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_eye.xml')

            cam=cv2.VideoCapture(0);
            while(True):
                #camera read
                ret,img=cam.read();
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces=faceDetect.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=7, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE);

                for(x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]
                    id,conf=face_recognizer.predict(gray[y:y+h,x:x+w])
                    profile=getProfile(id)
                    #print("[INFO] accurency: "+str(conf))
                    #detect ra mắt ở trên khuôn mặt đó
                    eyes = eye_cascade.detectMultiScale(roi_gray,scaleFactor=1.1, minNeighbors=10, minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
                    for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    #set text to window
                    if(profile!=None):
                        #cv2.PutText(cv2.fromarray(img),str(id),(x+y+h),font,(0,0,255),2);
                        cv2.putText(img, "Name: " + str(profile[1]), (x,y+h+25), fontface, fontscale, fontcolor ,2)
                        cv2.putText(img, "Age: " + str(profile[2]), (x,y+h+45), fontface, fontscale, fontcolor ,2)
                        cv2.putText(img, "Gender: " + str(profile[3]), (x,y+h+65), fontface, fontscale, fontcolor ,2)

                    cv2.imshow('Face',img)

                if cv2.waitKey(1)==ord('q'):
                    break;
            cam.release()
            cv2.destroyAllWindows()

        else:
            print("[INFO] Finish!")
            break
