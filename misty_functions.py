from mistyPy.Robot import Robot
from time import sleep as delay
import numpy as np
import cv2
import time
import base64

misty = Robot("192.168.1.237")
# print(current_response.json())


def move_head_no(misty, center_pitch):
    yaw_no = 20
    yaw_delay = 0.5
    yaw_duration = 0.1
    # Animate Misty to shake it's head
    misty.move_head(center_pitch, 0, -yaw_no, duration=yaw_duration)
    delay(yaw_delay)
    misty.move_head(center_pitch, 0, yaw_no, duration=yaw_duration)
    delay(yaw_delay)
    misty.move_head(center_pitch, 0, -yaw_no, duration=yaw_duration)
    delay(yaw_delay)
    misty.move_head(center_pitch, 0, yaw_no,duration= yaw_duration)
    delay(yaw_delay)
    # move back to center
    misty.move_head(center_pitch, 0, 0, duration=yaw_duration)
    return


def move_head_yes(misty, center_pitch):
    # Animate Misty to nod it's head
    pitch_yes = 10
    pitch_delay = 0.3
    pitch_speed = 95
    pitch_duration = 0.1

    # cap the pitch to prevent Misty from looking too far up or down
    move_down = np.clip((center_pitch + pitch_yes), a_min=-40, a_max=25).astype(float)
    move_up = np.clip((move_down - 2 * pitch_yes), a_min=-40, a_max=25).astype(float)

    curr_response = misty.move_head(move_up, 0, 0, duration=pitch_duration)
    print(f"moving head up to: {move_up}")
    delay(pitch_delay)

    curr_response = misty.move_head(move_down, 0, 0, duration=pitch_duration)
    print(f"moving head down to: {move_down}")
    delay(pitch_delay)

    curr_response = misty.move_head(move_up, 0, 0, duration=pitch_duration)
    print(f"moving head up to: {move_up}")
    delay(pitch_delay)

    curr_response = misty.move_head(move_down, 0, 0, duration=pitch_duration)
    print(f"moving head down to: {move_down}")
    delay(pitch_delay)

    # move back to center
    curr_response = misty.move_head(center_pitch, 0, 0, duration=pitch_duration)
    print("moving head back to center")
    return
def move_head_backchanneling(misty, center_pitch):
    yaw_left = -10  # Slight tilt to the left
    yaw_right = 10  # Slight tilt to the right
    yaw_delay = 0.5
    yaw_duration = 0.1
    
    # Slight head tilt to the left
    misty.move_head(center_pitch, yaw_left, 0, duration=yaw_duration)
    delay(yaw_delay)
    
    # Slight nod
    misty.move_head(center_pitch + 5, 0, 0, duration=yaw_duration)  # Slight upward nod
    delay(yaw_delay)
    
    # Head back to center position after nod
    misty.move_head(center_pitch, 0, 0, duration=yaw_duration)
    delay(yaw_delay)
    
    # Slight tilt to the left and back to center for backchanneling effect
    misty.move_head(center_pitch, yaw_left, 0, duration=yaw_duration)
    delay(yaw_delay)
    
    # Nod again
    misty.move_head(center_pitch + 5, 0, 0, duration=yaw_duration)
    delay(yaw_delay)
    
    # Move back to center
    misty.move_head(center_pitch, 0, 0, duration=yaw_duration)
    return

def move_head_backchanneling2(misty, center_pitch):
    yaw_left = -10  # Slight tilt to the left
    yaw_right = 10  # Slight tilt to the right
    yaw_delay = 0.5
    yaw_duration = 0.1
    
    # Slight head tilt to the left
    misty.move_head(center_pitch, yaw_right, 0, duration=yaw_duration)
    delay(yaw_delay)
    
    # Slight nod
    misty.move_head(center_pitch + 5, 0, 0, duration=yaw_duration)  # Slight upward nod
    delay(yaw_delay)
    
    # Head back to center position after nod
    misty.move_head(center_pitch, 0, 0, duration=yaw_duration)
    delay(yaw_delay)
    
    # Slight tilt to the left and back to center for backchanneling effect
    misty.move_head(center_pitch, yaw_right, 0, duration=yaw_duration)
    delay(yaw_delay)
    
    # Nod again
    misty.move_head(center_pitch + 5, 0, 0, duration=yaw_duration)
    delay(yaw_delay)
    
    # Move back to center
    misty.move_head(center_pitch, 0, 0, duration=yaw_duration)
    return


def play_audio(self, fileName, volume) -> bool:
    """Plays an audio file on Misty robot."""
    volume = int(volume)
    curr_response = self.play_audio(fileName=fileName, volume=volume)
    print(curr_response)
    return curr_response.status_code == 200

def upload_audio_to_misty(misty, file_path):
    # Read the audio file and convert to base64
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Upload to Misty
    response = misty.save_audio(
        fileName="intro.wav",
        data=base64_audio,
        immediatelyApply=True,
        overwriteExisting=True
    )
    
    print(f"Upload response: {response.status_code}")
    return response.status_code == 200


if __name__ == "__main__":
    eye_level_pitch = -5  # change this to the pitch of the eyes from face detection
    response = misty.get_known_faces()
    print(response.json())

    # misty.move_head(eye_level_pitch, 0, 0, 100)
    # This example sets Misty up to act as her own media server. Connect
    # to this stream from a client on the same network as Misty. The URL
    # for this stream would be: rtsp://<robot-ip-address>:1936

    # misty.start_av_streaming("rtsp:1930", 640, 480)
    # misty.start_video_streaming(5678,0,0,0,100,"false")
    # misty.stop_video_streaming()
    # misty.stop_av_streaming()
    # resp = misty.camera_service_enabled()
    # print(resp.json())
    # misty.disable_camera_service()
    # misty.enable_camera_service()
    # cap = cv2.VideoCapture(rtsp_url)

    # # Display the video stream
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Failed to retrieve frame.")
    #         break

    #     # Show the video frame
    #     cv2.imshow("RTSP Stream", frame)

    #     # Exit when 'q' is pressed
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # Release the resources
    # cap.release()
    # cv2.destroyAllWindows()

    # move_head_no(eye_level_pitch)

    # delay(1)
    # move_head_yes(eye_level_pitch)

    # misty.start_video_streaming(5678,0,0,0,100,"false")

