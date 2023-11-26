import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import streamlit as st

# Function to find face encodings from images
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

# Function to mark attendance
def mark_attendance(name, present_students):
    now = datetime.now()
    dt_string = now.strftime('%H:%M:%S')
    new_entry = {'Name': name, 'Time': dt_string}

    if name not in present_students:
        present_students.append(new_entry)

    return present_students

# Load training images
path = 'C:/Users/diksh/Desktop/attendance system/Training_images'
images = [cv2.imread(os.path.join(path, cl)) for cl in os.listdir(path)]
class_names = [os.path.splitext(cl)[0] for cl in os.listdir(path)]
encode_list_known = find_encodings(images)

# Streamlit UI
st.title("Attendance system using Face Recognition")

# Initialize session state
if 'present_students' not in st.session_state:
    st.session_state.present_students = []

# Start and Stop Buttons
start_recognition = st.sidebar.button("Start Recognition", key="start_button")
stop_recognition = st.sidebar.button("Stop Recognition", key="stop_button")

# Main loop
if start_recognition:
    success, img = cv2.VideoCapture(0).read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Face recognition code
    faces_cur_frame = face_recognition.face_locations(imgS)
    encodes_cur_frame = face_recognition.face_encodings(imgS, faces_cur_frame)

    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(encode_list_known, encode_face)
        match_index = np.argmin(face_dis)

        if matches[match_index]:
            name = class_names[match_index].upper()
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Mark attendance
            st.session_state.present_students = mark_attendance(name, st.session_state.present_students)

    # Display attendance on the Streamlit sidebar
    st.sidebar.write("Present Students:")
    for entry in st.session_state.present_students:
        st.sidebar.write(f"{entry['Name']} - {entry['Time']}")

    # Display the image with annotations
    st.image(img, channels="BGR")

# Display attendance after stopping recognition
if stop_recognition:
    st.sidebar.write("Present Students :")
    for entry in st.session_state.present_students:
        st.sidebar.write(f"{entry['Name']} - {entry['Time']}")
