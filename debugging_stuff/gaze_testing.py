import cv2
import mediapipe as mp
import time
from collections import deque
import numpy as np

class GazeTracker:
    def __init__(self, history_seconds=300, target_fps=15):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Store timestamps and gaze states for calculating engagement
        self.history_seconds = history_seconds
        self.gaze_history = deque(maxlen=history_seconds * 30)  # Assuming 30 fps
        self.start_time = None
        
        # Eye landmarks indices for mediapipe
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Define thresholds as class attributes
        self.EAR_THRESHOLD = 0.25
        self.VERTICAL_THRESHOLD = 15
        self.HORIZONTAL_THRESHOLD = 12

        self.MIN_FACE_SIZE = 0.15       # Minimum face size as proportion of frame
        self.CONFIDENCE_THRESHOLD = 0.8  # Minimum landmark confidence
        self.STABLE_FRAMES = 3          # Number of consecutive frames needed for positive detection
        
        # Storage for stability checking
        self.positive_detections = 0
        self.last_state = False

        # Target frame rate
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = time.time()
        
    def should_process_frame(self):
        current_time = time.time()
        if current_time - self.last_frame_time >= self.frame_interval:
            self.last_frame_time = current_time
            return True
        return False
        
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        points = []
        for i in eye_indices:
            points.append([landmarks[i].x, landmarks[i].y])
        points = np.array(points)
        
        # Calculate vertical distances
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])
        # Calculate horizontal distance
        h = np.linalg.norm(points[0] - points[3])
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h + 1e-6)  # Added small epsilon to prevent division by zero
        return ear
    
    def is_looking_at_robot(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        # Initialize default values
        is_looking = False
        left_ear = 0
        right_ear = 0
        x = 0
        y = 0
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark, self.LEFT_EYE)
            right_ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark, self.RIGHT_EYE)
            
            # Get head pose
            face_3d = []
            face_2d = []
            
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            
            # Camera matrix estimation
            focal_length = frame.shape[1]
            center = (frame.shape[1]/2, frame.shape[0]/2)
            cam_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            
            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix
            )
            
            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)
            
            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            
            # Get head direction
            y = angles[0] * 360
            x = angles[1] * 360
            
            # Normalize angles to be between -180 and 180
            x = (x + 180) % 360 - 180
            y = (y + 180) % 360 - 180
            
            # Check if person is looking at robot with stricter conditions
            is_looking = (
                left_ear > self.EAR_THRESHOLD and 
                right_ear > self.EAR_THRESHOLD and
                abs(y) < self.VERTICAL_THRESHOLD and 
                abs(x) < self.HORIZONTAL_THRESHOLD
            )
            
            #Draw face mesh (optional - comment out if you don't want the dots)
            for landmark in face_landmarks.landmark:
                x_px = int(landmark.x * frame.shape[1])
                y_px = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x_px, y_px), 1, (0, 255, 0), -1)
        
        # Add visual indicators regardless of face detection
        status_color = (0, 255, 0) if is_looking else (0, 0, 255)
        cv2.putText(frame, f"Looking: {is_looking}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        cv2.putText(frame, f"X: {x:.1f} Y: {y:.1f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                   
        cv2.putText(frame, f"Left EAR: {left_ear:.2f} Right EAR: {right_ear:.2f}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                   
        cv2.putText(frame, f"Thresholds - H: ±{self.HORIZONTAL_THRESHOLD}° V: ±{self.VERTICAL_THRESHOLD}°", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        return is_looking, frame
    
    def process_frame(self, frame):
        if self.start_time is None:
            self.start_time = time.time()
            
        is_looking, processed_frame = self.is_looking_at_robot(frame)
        self.gaze_history.append((time.time(), is_looking))
        
        return processed_frame
    
    def get_engagement_metrics(self):
        if len(self.gaze_history) < 2:
            return {
                'engagement_percentage': 0.0,
                'average_look_duration': 0.0,
                'look_count': 0
            }
            
        total_time = max(0.001, self.gaze_history[-1][0] - self.gaze_history[0][0])
        looking_frames = sum(1 for _, is_looking in self.gaze_history if is_looking)
        looking_time = (looking_frames / len(self.gaze_history)) * total_time
        
        look_durations = []
        current_duration = 0
        last_state = False
        
        for _, is_looking in self.gaze_history:
            if is_looking:
                current_duration += 1
            elif last_state:
                if current_duration > 0:
                    look_durations.append(current_duration)
                current_duration = 0
            last_state = is_looking
            
        if current_duration > 0:
            look_durations.append(current_duration)
            
        avg_duration = np.mean(look_durations) if look_durations else 0.0
            
        return {
            'engagement_percentage': (looking_time / total_time) * 100,
            'average_look_duration': avg_duration,
            'look_count': len(look_durations)
        }

def main():
    # Initialize camera
    cap = cv2.VideoCapture(5)  # Change this number to match your RealSense RGB camera
    
    if not cap.isOpened():
        print("Error: Could not open camera. Please check the camera index.")
        return
        
    tracker = GazeTracker()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            if tracker.should_process_frame():
                processed_frame = tracker.process_frame(frame)
                metrics = tracker.get_engagement_metrics()
                
                cv2.putText(processed_frame, 
                           f"Engagement: {metrics['engagement_percentage']:.1f}%",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Gaze Tracking', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
