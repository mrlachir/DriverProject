import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
from playsound import playsound
import threading

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance 1
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance 2
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)  # EAR formula
    return ear

# Load dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indices (for 68-point model)
L_EYE = list(range(36, 42))  # Left eye
R_EYE = list(range(42, 48))  # Right eye

# Constants
EAR_THRESHOLD = 0.25  # EAR below this means eyes are closed
CLOSED_EYES_DURATION = 5.0  # 5 seconds of closure before sound

def play_alert_sound():
    try:
        playsound("alert.wav")  # Path to your sound file
    except Exception as e:
        print(f"Error playing sound: {e}")

# Start webcam
cap = cv2.VideoCapture(0)

eyes_closed_start = None  # Tracks when eyes first close
alert_playing = False  # Prevents multiple sounds at once
ear = 0.0  # Default EAR value when no face is detected

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot access webcam!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    eyes_closed = False  # Reset each frame

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        left_eye = landmarks[L_EYE]
        right_eye = landmarks[R_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0  # Update ear when face is detected

        # Eyes are closed if EAR is below threshold
        if ear < EAR_THRESHOLD:
            eyes_closed = True

        # Optional: Draw eye landmarks
        for point in left_eye:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
        for point in right_eye:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

    # Timing logic for sound and alert text
    current_time = time.time()
    alert_text = ""  # Default: no alert text

    if eyes_closed:
        if eyes_closed_start is None:
            eyes_closed_start = current_time  # Start timer when eyes close
        elif (current_time - eyes_closed_start >= CLOSED_EYES_DURATION):
            alert_text = "Alert: Eyes closed for more than 5s"
            if not alert_playing:
                threading.Thread(target=play_alert_sound).start()
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