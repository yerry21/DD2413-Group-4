import cv2
import mediapipe as mp
import time
import numpy as np
import websocket
import threading
from collections import deque
from io import BytesIO
from PIL import Image
from mistyPy.Robot import Robot
from misty_functions import center_head_on_centroid
misty = Robot("192.168.1.237")

# For getting video input from Misty and process it remotely.

# You can tune the EAR and ver/hor thresholds accordingly
class GazeTracker:
    def __init__(self, history_seconds=300, target_fps=15):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.history_seconds = history_seconds
        self.gaze_history = deque(maxlen=history_seconds * 30)
        self.start_time = None
        
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        self.EAR_THRESHOLD = 0.15
        self.VERTICAL_THRESHOLD = 7
        self.HORIZONTAL_THRESHOLD = 5
        self.MIN_FACE_SIZE = 0.15
        self.CONFIDENCE_THRESHOLD = 0.8
        self.STABLE_FRAMES = 3
        
        self.positive_detections = 0
        self.last_state = False
        
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
        
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])
        h = np.linalg.norm(points[0] - points[3])
        
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear
    def calculate_eye_center(self,landmarks, eye_indices, frame_width, frame_height):
        """
        Calculate the mean position (center) of the eye landmarks.
        
        Args:
            landmarks: List of Mediapipe landmarks.
            eye_indices: Indices of the eye landmarks.
            frame_width: Width of the video frame.
            frame_height: Height of the video frame.
        
        Returns:
            (center_x, center_y): Mean position of the eye in pixel coordinates.
        """
        total_x, total_y = 0, 0
        for idx in eye_indices:
            total_x += landmarks[idx].x * frame_width  # Convert normalized to pixel
            total_y += landmarks[idx].y * frame_height  # Convert normalized to pixel

        center_x = total_x / len(eye_indices)
        center_y = total_y / len(eye_indices)
        
        return int(center_x), int(center_y)
    def is_looking_at_robot(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        is_looking = False
        left_ear = 0
        right_ear = 0
        x = 0
        y = 0
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            left_ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark, self.LEFT_EYE)
            right_ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark, self.RIGHT_EYE)
            
            face_3d = []
            face_2d = []
            
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            
            focal_length = frame.shape[1]
            center = (frame.shape[1] / 2, frame.shape[0] / 2)
            cam_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            
            y = angles[0] * 360
            x = angles[1] * 360
            
            x = (x + 180) % 360 - 180
            y = (y + 180) % 360 - 180
            
            is_looking = (
                left_ear > self.EAR_THRESHOLD and 
                right_ear > self.EAR_THRESHOLD and
                abs(y) < self.VERTICAL_THRESHOLD and 
                abs(x) < self.HORIZONTAL_THRESHOLD
            )
            
            for landmark in face_landmarks.landmark:
                x_px = int(landmark.x * frame.shape[1])
                y_px = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x_px, y_px), 1, (0, 255, 0), -1)

            
                

                
            # Calculate eye centers
            left_eye_center = self.calculate_eye_center(face_landmarks.landmark, self.LEFT_EYE, frame.shape[1], frame.shape[0])
            right_eye_center = self.calculate_eye_center(face_landmarks.landmark, self.RIGHT_EYE, frame.shape[1], frame.shape[0])
            
            # Draw the centers
            cv2.circle(frame, left_eye_center, 5, (255, 0, 0), -1)  # Green dot for left eye
            cv2.circle(frame, right_eye_center, 5, (255, 0, 0), -1)  # Blue dot for right eye
            
            print(f"Left Eye Center: {left_eye_center}, Right Eye Center: {right_eye_center}")
            
            eye_centroid = np.array([int(left_eye_center[0] + right_eye_center[0])/2,int(left_eye_center[1] + right_eye_center[1])/2 ])
            
            # cv2.circle(frame, eye_centroid, 5, (255, 0, 0), -1)  # Blue dot for right eye
                        # Compute offsets
            frame_center_x = frame.shape[1] / 2
            frame_center_y = frame.shape[0] / 2
            x_offset = eye_centroid[0] - frame_center_x
            y_offset = eye_centroid[1]- frame_center_y
            # After y_offset is calculated:
            # if natural_gaze_bool :
            y_tolerance = 10
            if abs(y_offset) > y_tolerance or abs(x_offset) > x_tolerance:
              center_head_on_centroid(misty,x_offset,y_offset,pitch_step=11,yaw_step=11)
            x_tolerance = 10




        
        status_color = (0, 255, 0) if is_looking else (0, 0, 255)
        cv2.putText(frame, f"Looking: {is_looking}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"X: {x:.1f} Y: {y:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"Left EAR: {left_ear:.2f} Right EAR: {right_ear:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
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
            return {'engagement_percentage': 0.0, 'average_look_duration': 0.0, 'look_count': 0}
        
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
            
        return {'engagement_percentage': (looking_time / total_time) * 100, 'average_look_duration': avg_duration, 'look_count': len(look_durations)}

# WebSocket video frame receiver
frame_queue = deque(maxlen=30)

def on_message(ws, message):
    img = Image.open(BytesIO(message))
    #frame = np.array(img)
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    frame_queue.append(frame)
    print(frame.shape)

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket connection opened")

def start_websocket_stream():
    ws = websocket.WebSocketApp("ws://192.168.1.237:5678", 
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()

# Start WebSocket server in a separate thread
threading.Thread(target=start_websocket_stream, daemon=True).start()

# Create GazeTracker instance
gaze_tracker = GazeTracker()

while True:
    if len(frame_queue) > 0:
        frame = frame_queue[-1]
        if gaze_tracker.should_process_frame():
            processed_frame = gaze_tracker.process_frame(frame)
            
            metrics = gaze_tracker.get_engagement_metrics()
            print(f"Engagement Percentage: {metrics['engagement_percentage']:.2f}%")
            print(f"Average Look Duration: {metrics['average_look_duration']:.2f} frames")
            print(f"Look Count: {metrics['look_count']}")

            cv2.putText(processed_frame, 
                           f"Engagement: {metrics['engagement_percentage']:.1f}%",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            display_frame = cv2.resize(frame, (600, 800))
            cv2.imshow('Processed Frame', display_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
