import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import pyttsx3
import queue

class VoiceAlert:
    def __init__(self):
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        for voice in voices:
            # Simple heuristic for male voice: 'david' or 'male'
            if "david" in voice.name.lower() or "male" in voice.name.lower() or "david" in voice.id.lower():
                self.engine.setProperty('voice', voice.id)
                break
        else:
            if voices:
                self.engine.setProperty('voice', voices[0].id)
        
        # Increase speech rate slightly for urgency
        self.engine.setProperty('rate', 170)
        
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
    def _worker(self):
        while True:
            text = self.q.get()
            self.engine.say(text)
            self.engine.runAndWait()
            self.q.task_done()
            
    def alert(self, text):
        if self.q.empty():
            self.q.put(text)

speaker = VoiceAlert()

# Mediapipe Face Mesh (Deep Learning Model using the New Tasks API)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.FaceLandmarker.create_from_options(options)

# 6 landmarks per eye based on standard Eye Aspect Ratio (EAR) layout
LEFT_EYE_INDICES = [362, 385, 386, 263, 374, 380] # Inner, Top1, Top2, Outer, Bottom1, Bottom2
RIGHT_EYE_INDICES = [33, 159, 158, 133, 145, 153]

# Variables suitable for detection
EAR_THRESHOLD = 0.22
CONSECUTIVE_FRAMES = 20  # ~0.6 seconds depending on camera FPS

# Face Movement / Yawn detection
MAR_LIP_INDICES = [13, 14, 78, 308] # Top, Bottom, Left Corner, Right Corner
MAR_THRESHOLD = 0.5
YAWN_CONSECUTIVE_FRAMES = 15

DISTRACTION_CONSECUTIVE_FRAMES = 20
frame_counter = 0
yawn_counter = 0
distraction_counter = 0

def calculate_ear(landmarks, eye_indices, width, height):
    """Calculates Eye Aspect Ratio (EAR) to determine if eyes are open or closed."""
    # Retrieve coordinates scaled to frame dimensions
    p = []
    for idx in eye_indices:
        lm = landmarks[idx]
        p.append(np.array([lm.x * width, lm.y * height]))
        
    p1, p2, p3, p4, p5, p6 = p

    # Vertical eye landmarks
    dist_v1 = np.linalg.norm(p2 - p6)
    dist_v2 = np.linalg.norm(p3 - p5)

    # Horizontal eye landmarks
    dist_h = np.linalg.norm(p1 - p4)

    # Calculate and return EAR
    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return ear

def calculate_mar(landmarks, indices, width, height):
    """Calculates Mouth Aspect Ratio (MAR) to detect yawning and face movement."""
    top = np.array([landmarks[indices[0]].x * width, landmarks[indices[0]].y * height])
    bottom = np.array([landmarks[indices[1]].x * width, landmarks[indices[1]].y * height])
    left = np.array([landmarks[indices[2]].x * width, landmarks[indices[2]].y * height])
    right = np.array([landmarks[indices[3]].x * width, landmarks[indices[3]].y * height])
    
    dist_v = np.linalg.norm(top - bottom)
    dist_h = np.linalg.norm(left - right)
    
    return dist_v / dist_h if dist_h > 0 else 0.0

def check_head_pose(landmarks, width, height):
    """Estimates head direction using 2D face landmarks."""
    nose = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    chin = landmarks[152]
    forehead = landmarks[10]

    left_dist = max(nose.x - left_cheek.x, 0.001)
    right_dist = max(right_cheek.x - nose.x, 0.001)
    
    top_dist = max(nose.y - forehead.y, 0.001)
    bottom_dist = max(chin.y - nose.y, 0.001)

    direction = "Forward"
    if left_dist / right_dist > 2.0:
        direction = "Right"
    elif right_dist / left_dist > 2.0:
        direction = "Left"

    if top_dist / bottom_dist > 1.5:
        direction = "Down (Head Drop)"
    elif bottom_dist / top_dist > 1.5:
        direction = "Up"
            
    return direction

def main():
    global frame_counter, yawn_counter, distraction_counter

    try:
        # Open the primary webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open live camera.")
            return

        print("Started Drowsiness Detection System with Live Camera!")
        print("Press 'q' or 'ESC' on the video window to quit.")

        while True:
            success, frame = cap.read()
            if not success:
                print("Loop exited: cap.read() returned False")
                break

            # Flip horizontally for selfie-view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert to RGB since mediapipe needs RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image with the DL Face Mesh model
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = detector.detect(mp_image)

            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    # Calculate EAR for both eyes
                    left_ear = calculate_ear(face_landmarks, LEFT_EYE_INDICES, w, h)
                    right_ear = calculate_ear(face_landmarks, RIGHT_EYE_INDICES, w, h)
                    
                    # Average EAR
                    ear = (left_ear + right_ear) / 2.0

                    # Calculate Mouth Aspect Ratio (MAR)
                    mar = calculate_mar(face_landmarks, MAR_LIP_INDICES, w, h)

                    # Draw eye landmarks for visualization
                    for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MAR_LIP_INDICES:
                        lm = face_landmarks[idx]
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)

                    # Display EAR and MAR values
                    cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    direction = check_head_pose(face_landmarks, w, h)
                    cv2.putText(frame, f"Pose: {direction}", (30, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Drowsiness Logic (Eyes shutting)
                    if ear < EAR_THRESHOLD:
                        frame_counter += 1
                        if frame_counter >= CONSECUTIVE_FRAMES:
                            speaker.alert("Drowsiness Detected!")
                            cv2.putText(frame, "** DROWSINESS ALERT! **", (w//2 - 200, h//2 - 40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
                    else:
                        frame_counter = 0

                    # Yawning / Face Movement Logic
                    if mar > MAR_THRESHOLD:
                        yawn_counter += 1
                        if yawn_counter >= YAWN_CONSECUTIVE_FRAMES:
                            speaker.alert("Yawning Detected!")
                            cv2.putText(frame, "** YAWNING DETECTED! **", (w//2 - 200, h//2 + 40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 4)
                    else:
                        yawn_counter = 0

                    # Distraction / Head Direction Logic
                    if direction != "Forward":
                        distraction_counter += 1
                        if distraction_counter >= DISTRACTION_CONSECUTIVE_FRAMES:
                            speaker.alert("Please look forward")
                            cv2.putText(frame, "** DISTRACTION WARNING! **", (w//2 - 200, h//2 + 120), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 4)
                    else:
                        distraction_counter = 0

            # Show the video feed
            cv2.imshow("AI Drowsiness Detection System", frame)

            # Quit on 'ESC' or 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("Loop exited: ESC key pressed")
                break
            elif key == ord('q'):
                print("Loop exited: 'q' key pressed")
                break
            
            # Check if GUI window is closed by clicking X button
            # (getWindowProperty returns -1 if the window is closed)
            if cv2.getWindowProperty("AI Drowsiness Detection System", cv2.WND_PROP_VISIBLE) < 1:
                print("Loop exited: Window closed manually")
                break

    except Exception as e:
        import traceback
        print("An error occurred during execution:")
        traceback.print_exc()
    finally:
        # Cleanup
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed, exiting.")

if __name__ == "__main__":
    main()
