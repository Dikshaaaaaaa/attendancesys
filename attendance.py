import cv2
import face_recognition
import os
from datetime import datetime
import csv
import boto3


path = r'C:\Users\diksh\Desktop\New folder (4)\Training_images'
images = [cv2.imread(f'{path}/{cl}') for cl in os.listdir(path)]
classNames = [os.path.splitext(cl)[0] for cl in os.listdir(path)]


encodeListKnown = [face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0] for img in images]


csv_path = r'C:\Users\diksh\Desktop\New folder (4)\Attendance.csv'


aws_access_key_id = 'AKIAUICQMHXDEIUZL6JN'
aws_secret_access_key = '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'
bucket_name = 'saswfr'

s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

def upload_to_s3(file_path, bucket_name, object_name):
    try:
        s3.upload_file(file_path, bucket_name, object_name)
        print(f"File uploaded successfully to {bucket_name}/{object_name}")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")

def markAttendance(name, present_students):
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d')
    csv_path_with_date = fr'C:\Users\diksh\Desktop\New folder (4)\Attendance_{date_string}.csv'
    
    if not os.path.isfile(csv_path_with_date):
        with open(csv_path_with_date, 'w', newline='') as f:
            f.write('Date,Name,Time\n')

    existing_data = set()
    with open(csv_path_with_date, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  
        for row in reader:
            existing_data.add((row[0], row[1], row[2]))

    dtString = now.strftime('%H:%M:%S')
    new_entry = (date_string, name, dtString)
    
    if new_entry not in existing_data and name not in present_students:
        with open(csv_path_with_date, 'a', newline='') as f:
            f.write(f'{date_string},{name},{dtString}\n')
        present_students.add(name)

        
        upload_to_s3(csv_path_with_date, bucket_name, f"Attendance/{os.path.basename(csv_path_with_date)}")

    return present_students

cap = cv2.VideoCapture(0)
present_students = set()

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = min(range(len(faceDis)), key=faceDis.__getitem__)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            present_students = markAttendance(name, present_students)

    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
