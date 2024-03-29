# age_gender.py

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

class AgeGenderEstimator:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.face_size = (224, 224)  # Adjust this size according to your model's input size

    def preprocess_input(self, face):
        # Preprocess the face image before feeding it to the model
        # You may need to adjust preprocessing steps based on your model's requirements
        face = cv2.resize(face, self.face_size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = np.expand_dims(face, axis=-1)
        face = face / 255.0  # Normalize to [0, 1]
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        return face

    def predict(self, face):
        # Preprocess the input face image
        preprocessed_face = self.preprocess_input(face)

        # Predict age and gender
        age_gender_predictions = self.model.predict(preprocessed_face)

        age = np.argmax(age_gender_predictions[0][:100])  # Assuming age prediction output has 100 classes
        gender_prob = age_gender_predictions[0][100]  # Assuming gender prediction output has 2 classes

        # Determine gender based on the predicted probability
        gender = 'Male' if gender_prob > 0.5 else 'Female'

        return gender, age

if __name__ == '__main__':
    # Sample code to test the AgeGenderEstimator
    model_path = 'path/to/age_gender_model.h5'  # Specify the correct path to your pre-trained model
    age_gender_estimator = AgeGenderEstimator(model_path)

    # Load a sample face image
    sample_face = cv2.imread('sample_face.jpg')  # Provide the path to a sample face image

    # Predict age and gender for the sample face
    gender, age = age_gender_estimator.predict(sample_face)

    print(f'Predicted Gender: {gender}')
    print(f'Predicted Age: {age}')
