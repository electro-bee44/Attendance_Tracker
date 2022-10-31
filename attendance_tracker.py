import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images'
images = []
personName = []
myList = os.listdir(path)
print(myList)
for current_img in myList:
    current_IMG = cv2.imread(f'{path}/{current_img}')
    images.append(current_IMG)
    personName.append(os.path.splitext(current_img)[0])
print(personName)


def faceEncodings(images1):
    encodelist = []
    for img in images1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


encodelistknown = faceEncodings(images)
print("Encodings Completed")


def attendance(name):
    with open('attendance.csv', 'r+') as f:
        my_datalist = f.readlines()
        name_list = []

        for line in my_datalist:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            time_now = datetime.now()
            t_str = time_now.strftime('%H:%M:%S')
            d_str = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{t_str},{d_str}')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    faces_current_frame = face_recognition.face_locations(faces)
    encodescurrent_frame = face_recognition.face_encodings(faces, faces_current_frame)

    for encodeface, facelocation in zip(encodescurrent_frame, faces_current_frame):
        matches = face_recognition.compare_faces(encodelistknown, encodeface)
        faceDis = face_recognition.face_distance(encodelistknown, encodeface)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = facelocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
            attendance(name)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()
