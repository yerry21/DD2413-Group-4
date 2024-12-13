from mistyPy.Robot import Robot
from time import sleep as delay
import numpy as np
import cv2
import time
import base64



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


class MistyController:
    def __init__(self, ip="192.168.1.237"):
        """
        Initialize the MistyController with a Misty robot instance.
        
        :param misty: Misty robot instance
        """
        self.misty = Robot(ip)
        self.misty.start_video_streaming(5678, 90, 0, 0, 50, "false")
        
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


    def move_head_no(self, center_pitch):
        """
        Animate Misty to shake its head (no).
        
        :param center_pitch: Central pitch position
        """
        yaw_no = 20
        yaw_delay = 0.5
        yaw_duration = 0.1
        
        for _ in range(2):
            self.misty.move_head(center_pitch, 0, -yaw_no, duration=yaw_duration)
            time.sleep(yaw_delay)
            self.misty.move_head(center_pitch, 0, yaw_no, duration=yaw_duration)
            time.sleep(yaw_delay)
        
        # Move back to center
        self.misty.move_head(center_pitch, 0, 0, duration=yaw_duration)

    def move_head_yes(self, center_pitch):
        """
        Animate Misty to nod its head (yes).
        
        :param center_pitch: Central pitch position
        """
        pitch_yes = 10
        pitch_delay = 0.3
        pitch_duration = 0.1

        # Clip pitch to prevent looking too far up or down
        move_down = np.clip((center_pitch + pitch_yes), a_min=-40, a_max=25).astype(float)
        move_up = np.clip((move_down - 2 * pitch_yes), a_min=-40, a_max=25).astype(float)

        for _ in range(2):
            self.misty.move_head(move_up, 0, 0, duration=pitch_duration)
            time.sleep(pitch_delay)
            self.misty.move_head(move_down, 0, 0, duration=pitch_duration)
            time.sleep(pitch_delay)

        # Move back to center
        self.misty.move_head(center_pitch, 0, 0, duration=pitch_duration)

    def move_head_backchanneling(self, center_pitch, variant=1):
        """
        Animate Misty's head for backchanneling.
        
        :param center_pitch: Central pitch position
        :param variant: Backchanneling variant (1 or 2)
        """
        yaw_side = 10 if variant == 1 else -10
        yaw_delay = 0.5
        yaw_duration = 0.1
        
        # Slight head tilt
        self.misty.move_head(center_pitch, yaw_side, 0, duration=yaw_duration)
        time.sleep(yaw_delay)
        
        # Slight nod
        self.misty.move_head(center_pitch + 5, 0, 0, duration=yaw_duration)
        time.sleep(yaw_delay)
        
        # Head back to center
        self.misty.move_head(center_pitch, 0, 0, duration=yaw_duration)
        time.sleep(yaw_delay)

    def upload_audio(self, file_path, file_name="audio.wav"):
        """
        Upload an audio file to Misty.
        
        :param file_path: Path to the audio file
        :param file_name: Name to save the audio file as
        :return: Boolean indicating successful upload
        """
        # Read the audio file and convert to base64
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Upload to Misty
        response = self.misty.save_audio(
            fileName=file_name,
            data=base64_audio,
            immediatelyApply=True,
            overwriteExisting=True
        )
        
        print(f"Audio upload response: {response.status_code}")
        return response.status_code == 200

    def handle_audio_prompt(self, prompt):
        """
        Handle different audio prompts and play corresponding audio files.
        
        :param prompt: Numeric prompt identifying the current game state
        :return: Boolean indicating successful audio handling
        """
        # Check if prompt exists in our mapping
        if prompt not in self._prompt_audio_map:
            print(f"Unknown prompt: {prompt}")
            return False
        
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
            self.upload_audio("audios/" + audio)
            time.sleep(2)
            return self.handle_audio_prompt(4)

        if 12 <= prompt <= 16:
            # Play sorry audio for wrong animal answers
            self.upload_audio("audios/" + audio_sorry)
            time.sleep(2)        

        # Upload and play the appropriate audio
        return self.upload_audio("audios/" + audio)

    def center_head_on_centroid(self, x_offset, y_offset, pitch_step=10, yaw_step=10, x_tolerance=10, y_tolerance=10):
        """
        Adjust Misty's head to center on the eye centroid.
        
        :param x_offset: Horizontal offset of the eye centroid
        :param y_offset: Vertical offset of the eye centroid
        :param pitch_step: Amount to adjust pitch per iteration
        :param yaw_step: Amount to adjust yaw per iteration
        :param x_tolerance: Acceptable horizontal offset range
        :param y_tolerance: Acceptable vertical offset range
        """
        current_pitch = 0
        current_yaw = 0

        # Handle vertical adjustment (pitch)
        if abs(y_offset) > y_tolerance:
            current_pitch += pitch_step if y_offset > 0 else -pitch_step
            current_pitch = max(-40, min(25, current_pitch))
            self.misty.move_head(current_pitch, 0, current_yaw, duration=0.1)
            print(f"Adjusting Pitch: {current_pitch} degrees")

        # Handle horizontal adjustment (yaw)
        if abs(x_offset) > x_tolerance:
            current_yaw -= yaw_step if x_offset > 0 else -yaw_step
            current_yaw = max(-90, min(90, current_yaw))
            self.misty.move_head(current_pitch, 0, current_yaw, duration=0.1)
            print(f"Adjusting Yaw: {current_yaw} degrees")


if __name__ == "__main__":
    eye_level_pitch = 0  # change this to the pitch of the eyes from face detection
    # response = misty.get_known_faces()
    # print(response.json())

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

