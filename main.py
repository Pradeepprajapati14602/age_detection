# main.py

import cv2
from age_gender import AgeGenderEstimator

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('detection/data/haarcascade_frontalface_default.xml')

# Initialize the age and gender estimator
age_gender_estimator = AgeGenderEstimator(model_path='detection/data/age_gender_model.h5')

# Initialize counts
male_count = 0
female_count = 0

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Face detection using Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Perform gender classification and age estimation
        gender, age = age_gender_estimator.predict(face_img)

        if gender == 'Male':
            male_count += 1
        elif gender == 'Female':
            female_count += 1

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display age and gender predictions near the face
        text = f'Age: {age}, Gender: {gender}'
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display counts
    total_count = male_count + female_count
    cv2.putText(frame, f'Male: {male_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Female: {female_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Total: {total_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Gender and Age Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
