import cv2
import mediapipe as mp
import numpy as np

class GazeEngagementDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]  # Left eye landmarks
        self.RIGHT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]  # Right eye landmarks
        self.EAR_THRESHOLD = 0.2
        self.HORIZONTAL_THRESHOLD = 0.35
        self.VERTICAL_THRESHOLD = 0.35

    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        # Compute EAR (Eye Aspect Ratio)
        a = np.linalg.norm(np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) -
                           np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]))
        b = np.linalg.norm(np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]) -
                           np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]))
        c = np.linalg.norm(np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) -
                           np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]))
        return (a + b) / (2.0 * c)

    def calculate_pupil_position(self, landmarks, eye_indices, frame_shape):
        # Calculate normalized pupil position in [0, 1] relative to the eye bounding box
        points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
        bbox_x_min = np.min(points[:, 0]) * frame_shape[1]
        bbox_y_min = np.min(points[:, 1]) * frame_shape[0]
        bbox_x_max = np.max(points[:, 0]) * frame_shape[1]
        bbox_y_max = np.max(points[:, 1]) * frame_shape[0]
        
        bbox_width = bbox_x_max - bbox_x_min
        bbox_height = bbox_y_max - bbox_y_min
        
        # Get center of the eye
        eye_center_x = (points[:, 0].mean() * frame_shape[1]) - bbox_x_min
        eye_center_y = (points[:, 1].mean() * frame_shape[0]) - bbox_y_min
        
        # Normalize center to [0, 1] relative to bounding box
        norm_x = eye_center_x / bbox_width
        norm_y = eye_center_y / bbox_height
        
        return norm_x, norm_y

    def is_looking_at_robot(self, frame, results):
        if not results.multi_face_landmarks:
            return False

        face_landmarks = results.multi_face_landmarks[0]
        left_ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark, self.LEFT_EYE)
        right_ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark, self.RIGHT_EYE)

        left_pupil = self.calculate_pupil_position(face_landmarks.landmark, self.LEFT_EYE, frame.shape)
        right_pupil = self.calculate_pupil_position(face_landmarks.landmark, self.RIGHT_EYE, frame.shape)

        # Check gaze direction based on pupil position
        is_gaze_centered = (
            0.4 <= left_pupil[0] <= 0.6 and
            0.4 <= left_pupil[1] <= 0.6 and
            0.4 <= right_pupil[0] <= 0.6 and
            0.4 <= right_pupil[1] <= 0.6
        )

        # Combine head pose and gaze check
        is_engaged = (
            left_ear > self.EAR_THRESHOLD and
            right_ear > self.EAR_THRESHOLD and
            is_gaze_centered
        )

        return is_engaged

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        is_engaged = self.is_looking_at_robot(frame, results)
        status_color = (0, 255, 0) if is_engaged else (0, 0, 255)
        status_text = "Engaged" if is_engaged else "Not Engaged"

        # # Visualize results
        # if results.multi_face_landmarks:
        #     for face_landmarks in results.multi_face_landmarks:
        #         self.mp_face_mesh.drawing_utils.draw_landmarks(
        #             frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
        #             landmark_drawing_spec=self.mp_face_mesh.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                #)
        cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        return frame

def main():
    gaze_detector = GazeEngagementDetector()
    cap = cv2.VideoCapture(5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = gaze_detector.process_frame(frame)
        cv2.imshow('Gaze Engagement Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
