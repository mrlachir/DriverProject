# Driver Eye Monitor

Detects if your eyes are closed for 5+ seconds, plays a sound, and shows an alert.

---

## Setup
1. **Install Python**:
   - Download from [python.org](https://www.python.org/downloads/) (3.*).
   - Check "Add Python to PATH" during install.

2. **Install Libraries**:
   - Open Command Prompt, go to this folder:
     ```
     cd path\to\DriverProject
     ```
   - Run:
     ```
     pip install opencv-python numpy scipy dlib playsound
     ```

3. **Files Needed**:
   - `driver_monitor_dlib.py` (script)
   - `shape_predictor_68_face_landmarks.dat` (in folder)
   - `alert.wav` (in folder)

---

## Run It
1. In Command Prompt:
   ```
   cd path\to\DriverProject
   python driver_monitor_dlib.py
   ```
2. Close eyes for 5s to test sound and alert.
3. Press `q` to quit.


