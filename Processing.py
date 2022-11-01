# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:54:57 2021
@author: Cao Van Vinh
NHận diện khuôn mặt với OpenCV

"""

# Face Recognition with OpenCV
import cv2
import os
import numpy as np
import pickle
import sqlite3


def processImage(data_folder_path):
    subject_images_names = os.listdir(data_folder_path)
    # load OpenCV face detector use LBP  more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
    # detect face and add face to list of faces
    for image_name in subject_images_names:
        # build image path
        image_paths = data_folder_path+"/"+image_name
        subject_images_names1 = os.listdir(image_paths)
        sampleNum=0
        id = image_name
        for image_path in subject_images_names1:
            every_image = image_paths+"/"+image_path

            # read image
            image = cv2.imread(every_image)

            # display an image window to show the image
            cv2.imshow("Processing on image...",cv2.resize(image,(400,500)))
            cv2.waitKey(1000)

            # detect face
            # convert the test image to gray image as opencv face detector expects gray images
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # detect multiscale images
            faces = face_cascade.detectMultiScale(gray, 1.1,5)  # result is a list of faces

            for (x,y,w,h) in faces:
                everygray = gray[y:y+h,x:x+w]
                cv2.rectangle(everygray,(x,y),(x+w,y+h),(255,0,0),2)
                #incrementing sample number
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder
                cv2.imwrite("dataSet/User."+id +'.'+ str(sampleNum) + ".jpg", everygray)

                cv2.imshow('Face',everygray)
        print("folder "+str(image_name)+": "+str(sampleNum)+" image processed.");

        #wait for 100 miliseconds
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

print("Processing data...")
processImage("ImgExternal")

print("Data processed!")








