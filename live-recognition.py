import numpy as np
import cv2
import copy
import sys
import csv
import ast
import os
from initiate_model import initiate_model

# The argument passed in when launching this script should be the name of the model 
# the user wants to use for the facial expression recognition task.
name = sys.argv[1]
parameter_csv = 'parameters.csv'
model_weights_file = name + '_weights.h5'
param_d = dict()

with open(parameter_csv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader)
        for row in reader:
            row_data = []
            for field in row:
                try:
                    field = ast.literal_eval(field)
                except ValueError:
                    pass
                row_data.append(field)
            param_d[row_data[0]] = row_data

# print(param_d)
if name not in param_d:
     print("The model you specified does not exist in our database")
     quit()
if not os.path.exists(model_weights_file):
     print("The model you specified exists in our database but it has not been trained")
     quit()
model = initiate_model(*(param_d[name][1:]))
model.load_weights(model_weights_file)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 800, 600)

while True:
    
    ret, frame = cap.read()
    img = copy.deepcopy(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        fc = gray[y:y+h, x:x+w]
        
        roi = cv2.resize(fc, (56, 56))
        pred = model.predict(roi[np.newaxis, :, :, np.newaxis])
        text_idx=np.argmax(pred)
        text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        text= text_list[text_idx]
        cv2.putText(img, text, (x, y-5),
           cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 255), 2)
        img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
            
    
    cv2.imshow("frame", img)
    key = cv2.waitKey(1) & 0xFF
    if key== ord('q'):
        break
    if key == ord('s'):
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 800, 600)
        cv2.imshow("image", img)
        cv2.waitKey()
        cv2.destroyWindow("image")
    
cap.release()
cv2.destroyAllWindows()