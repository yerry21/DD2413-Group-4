from mistyPy.Robot import Robot
from time import sleep as delay
import numpy as np
import cv2

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

# def center_head_on_centroid(misty, x_offset,y_offset, pitch_step=10, y_tolerance=10):
#     """
#     Adjust Misty's head pitch to center the eye centroid vertically.

#     Parameters:
#     - misty: Misty robot instance
#     - y_offset: Vertical offset of the eye centroid from the image center (pixels)
#     - pitch_step: Amount to adjust pitch per iteration (degrees)
#     - y_tolerance: Acceptable range for y_offset to be considered centered
#     """
#     # Current pitch starts at 0 (or an assumed starting value)
#     current_pitch = 0
#     print(y_offset)
#     if abs(y_offset) > y_tolerance:
#         if y_offset > 0:
#             # Eye centroid is below center - move head up
#             current_pitch += pitch_step
#         else:
#             # Eye centroid is above center - move head down
#             current_pitch -= pitch_step

#         # Clamp pitch within Misty's allowed range
#         current_pitch = max(-40, min(25, current_pitch))

#         # Move Misty's head
#         misty.move_head(current_pitch, 0, 0, duration=0.1)
#         print(f"Adjusting Pitch: {current_pitch} degrees")

        
#     else : 
#         print("Head Centered vertically!")
def center_head_on_centroid(misty, x_offset, y_offset, pitch_step=10, yaw_step=10, tolerance=10):
    """
    Adjust Misty's head yaw and pitch to center the eye centroid both vertically and horizontally.

    Parameters:
    - misty: Misty robot instance
    - x_offset: Horizontal offset of the eye centroid from the image center (pixels)
    - y_offset: Vertical offset of the eye centroid from the image center (pixels)
    - pitch_step: Amount to adjust pitch per iteration (degrees)
    - yaw_step: Amount to adjust yaw per iteration (degrees)
    - tolerance: Acceptable range for offsets to be considered centered (pixels)
    """

    # Initialize current pitch and yaw
    current_pitch = 0
    current_yaw = 0

    # Handle vertical adjustment (pitch)
    if abs(y_offset) > tolerance:
        if y_offset > 0:
            # Eye centroid is below center - move head down
            current_pitch += pitch_step
        else:
            # Eye centroid is above center - move head up
            current_pitch -= pitch_step

        # Clamp pitch to Misty's range (-40 to 25 degrees)
        current_pitch = max(-40, min(25, current_pitch))

        # Move Misty's head vertically
        misty.move_head(current_pitch, 0, current_yaw, duration=0.1)
        print(f"Adjusting Pitch: {current_pitch} degrees")
    else:
        print("Head centered vertically!")

    # Handle horizontal adjustment (yaw)
    if abs(x_offset) > tolerance:
        if x_offset > 0:
            # Eye centroid is to the right - move head right
            current_yaw += yaw_step
        else:
            # Eye centroid is to the left - move head left
            current_yaw -= yaw_step

        # Clamp yaw to Misty's range (-90 to 90 degrees)
        current_yaw = max(-90, min(90, current_yaw))

        # Move Misty's head horizontally
        misty.move_head(current_pitch, 0, current_yaw, duration=0.1)
        print(f"Adjusting Yaw: {current_yaw} degrees")
    else:
        print("Head centered horizontally!")


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

