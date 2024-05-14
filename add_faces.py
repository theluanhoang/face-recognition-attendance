import os
import pickle

import cv2
import numpy as np

video = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
faces_data = []

i = 0

name = input("Enter your name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) <= 100 and (i % 10 == 0):
            faces_data.append(resized_img)
        i = i + 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord("q") or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# FACES DATA BEFORE
with open('data/debug.txt', 'a') as f:
    f.write("============================== FACES DATA BEFORE ===============================\n")
    f.write("\n".join(map(str, faces_data)) + "\n")
    
faces_data=np.asarray(faces_data)

# FACES DATA AFTER - 1
with open('data/debug.txt', 'a') as f:
    f.write("============================== FACES DATA AFTER - 1 ===============================\n")
    f.write("\n".join(map(str, faces_data)) + "\n")

faces_data=faces_data.reshape(100, -1)

# FACES DATA AFTER - 2
with open('data/debug.txt', 'a') as f:
    f.write("============================== FACES DATA AFTER - 2 ===============================\n")
    f.write("\n".join(map(str, faces_data)) + "\n")


# NAMES.PKL
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
        
# FACES_DATA.PKL
if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)