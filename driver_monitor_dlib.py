import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
from playsound import playsound
import threading
import random

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indices
L_EYE = list(range(36, 42))
R_EYE = list(range(42, 48))

# Constants
EAR_THRESHOLD = 0.25
CLOSED_EYES_DURATION = 5.0

def play_alert_sound():
    try:
        playsound("alert.wav")
    except Exception as e:
        print(f"Error playing sound: {e}")

def simulate_vibration():
    # Simulate steering wheel vibration by shaking the window
    for _ in range(10):  # Shake 10 times
        x_offset = random.randint(-10, 10)  # Random horizontal shake
        y_offset = random.randint(-10, 10)  # Random vertical shake
        cv2.moveWindow("Driver Monitoring", 100 + x_offset, 100 + y_offset)
        time.sleep(0.05)  # Quick shake duration
    # Reset window position
    cv2.moveWindow("Driver Monitoring", 100, 100)

# Start webcam
cap = cv2.VideoCapture(0)
eyes_closed_start = None
alert_playing = False
ear = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot access webcam!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    eyes_closed = False

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        left_eye = landmarks[L_EYE]
        right_eye = landmarks[R_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESHOLD:
            eyes_closed = True

        for point in left_eye:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
        for point in right_eye:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

    # Timing logic
    current_time = time.time()
    alert_text = ""

    if eyes_closed:
        if eyes_closed_start is None:
            eyes_closed_start = current_time
        elif (current_time - eyes_closed_start >= CLOSED_EYES_DURATION) and not alert_playing:
            alert_text = "Alert: Eyes closed for more than 5s"
            threading.Thread(target=play_alert_sound).start()
            threading.Thread(target=simulate_vibration).start()
            alert_playing = True
    else:
        eyes_closed_start = None
        alert_playing = False

    # Display status
    status = "Eyes Closed" if eyes_closed else "Eyes Open"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if alert_text:
        cv2.putText(frame, alert_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Driver Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()