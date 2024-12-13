from mistyPy.Robot import Robot
from time import sleep as delay
import numpy as np
import cv2
import time
import base64

misty = Robot("192.168.1.237")
misty.start_video_streaming()
# print(current_response.json())

# Audio sets
audio_sets_animals = ["animal1.wav", "animal2.wav", "animal3.wav", "animal4.wav", "animal5.wav"]
audio_welcome = "intro.wav"
audio_sets_next_round = ["Ask.wav", "AskNew.wav", "WhatQuestion.wav"]
audio_sets_correct = ["YouGuessedIt.wav", "AbsolutelyRight.wav", "ThatsCorrect.wav"]
audio_sets_ask_guess = ["MakeAGuess.wav", "WannaMakeAGuess.wav", "TimeToMakeAGuess.wav"]
audio_finished = "ThanksForPlaying.wav"
audio_questions_rem = ["5left.wav", "4left.wav", "3left.wav", "2left.wav", "1left.wav", "0left.wav"]
audio_sorry = "sorry.wav"

class AudioHandler:
    def __init__(self):
        # Tracking counters for cycling audio sets
        self._counters = {
            'next_round': 0,
            'correct': 0,
            'ask_guess': 0
        }
        
        # Mapping of prompts to audio files or audio sets
        self._prompt_audio_map = {
            1: audio_welcome,  # Welcome
            2: audio_sets_next_round,  # Next round
            3: audio_sets_correct,  # Correct
            4: audio_sets_ask_guess,  # Ask to guess
            5: audio_finished,  # Finished
            6: audio_questions_rem[0],  # 5 questions left
            7: audio_questions_rem[1],  # 4 questions left
            8: audio_questions_rem[2],  # 3 questions left
            9: audio_questions_rem[3],  # 2 questions left
            10: audio_questions_rem[4],  # 1 question left
            11: audio_questions_rem[5],  # No questions left
            12: audio_sets_animals[0],  # Wrong answer for animal 1
            13: audio_sets_animals[1],  # Wrong answer for animal 2
            14: audio_sets_animals[2],  # Wrong answer for animal 3
            15: audio_sets_animals[3],  # Wrong answer for animal 4
            16: audio_sets_animals[4]   # Wrong answer for animal 5
        }
    
    def handle_prompt(self, misty, prompt):
        """
        Handle different prompts and play corresponding audio files.
        
        :param misty: Misty robot instance
        :param prompt: Numeric prompt identifying the current game state
        """
        # Check if prompt exists in our mapping
        if prompt not in self._prompt_audio_map:
            print(f"Unknown prompt: {prompt}")
            return
        
        audio_source = self._prompt_audio_map[prompt]
        
        # Handle cycling audio sets
        if prompt in [2, 3, 4]:
            # Determine which counter to use based on the prompt
            counter_key = {
                2: 'next_round',
                3: 'correct',
                4: 'ask_guess'
            }[prompt]
            
            # Cycle through the audio set
            self._counters[counter_key] += 1
            audio = audio_source[self._counters[counter_key] % len(audio_source)]
        else:
            # For non-cycling prompts, use the audio directly
            audio = audio_source

        if prompt == 11:
            # If no questions left, play the finished audio and ask to guess
            upload_audio_to_misty(misty, "audios/" + audio)
            time.sleep(2)
            return self.handle_prompt(misty, 4)

        if 12 <= prompt <= 16:
            upload_audio_to_misty(misty, "audios/" + audio_sorry)

            time.sleep(2)        
        # Upload audio to Misty
        return upload_audio_to_misty(misty, "audios/" + audio)
        


def start_streaming(misty):
    return misty.start_video_streaming(5678,90,0,0,50,"false")
def start_face_detection(misty):
    return misty.post_request("faces/detection/start")

def get_face_data(misty):
    return misty.get_request("faces/detection")

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
    return True

def center_head_on_centroid(misty, x_offset, y_offset, pitch_step=10, yaw_step=10, x_tolerance=10,y_tolerance=10):
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
    if abs(y_offset) > y_tolerance:
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
    if abs(x_offset) > x_tolerance:
        if x_offset > 0:
            # Eye centroid is to the left - move head left
            current_yaw -= yaw_step
        else:
            # Eye centroid is to the right - move head right
            current_yaw += yaw_step

        # Clamp yaw to Misty's range (-90 to 90 degrees)
        current_yaw = max(-90, min(90, current_yaw))

        # Move Misty's head horizontally
        misty.move_head(current_pitch, 0, current_yaw, duration=0.1)
        print(f"Adjusting Yaw: {current_yaw} degrees")
    else:
        print("Head centered horizontally!")


if __name__ == "__main__":
    eye_level_pitch = 0  # change this to the pitch of the eyes from face detection
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

