import cv2
import numpy as np
from tensorflow.keras.models import load_model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model2 = load_model(r"recognition_model.keras")
def webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        return
    while True:
        key = cv2.waitKey(1) & 0xFF
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = frame[y:y + 128, x:x + 128]
            input_frame = np.expand_dims(face/255, axis=0)
            prediction = model2.predict(input_frame)
            if prediction < 0.01:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(frame, "Nikhil", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),2, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Webcam", frame)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
webcam()
