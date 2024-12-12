from flask import Flask, request, jsonify
import threading
import time
from misty_functions import move_head_no, move_head_yes, play_audio, move_head_backchanneling, upload_audio_to_misty, start_streaming
from misty_functions import AudioHandler
from mistyPy.Robot import Robot
import numpy as np

DEBUG = True

app = Flask(__name__)

misty = Robot("192.168.1.237")
audio_handler  = AudioHandler()

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

guiData = {
    "ans": 0, # 0: no answer, 1: yes, 2: no, 3: maybe
    "delay_enabled": False,  # Delay
    "prompt" : 0
}

audio_sets_yes = ["yes.wav", "yeah.wav", "uh_huh.wav", "right.wav"]
audio_sets_no = ["no.wav", "no2.wav", "nah.wav", "sorry.wav"]
audio_sets_maybe = ["maybe.wav"]
audio_sets_backchannel = ["hmmmm.wav"]
audio_welcome = "intro.wav"


delay_duration = 1  # seconds
backchannel_chance = 0.99 # 50% chance of backchannel


@app.route("/")
def index():
    return open("htmls/misty_finalv3.html").read()


@app.route("/process", methods=["POST"])
def process():
    global guiData
    data = request.json
    print("Received data:", data)
    guiData = data
    return jsonify({"result": True})

def handle_answer(ans, delay):
    if ans == 1: # Yes
        print("Yes, delay : ", delay)
        if np.random.rand() < backchannel_chance: # chance of backchannel
                upload_audio_to_misty(misty, f"audios/{audio_sets_backchannel[np.random.randint(0, len(audio_sets_backchannel))]}")
                move_head_yes(misty, 0)
                upload_audio_to_misty(misty, f"audios/{audio_sets_yes[np.random.randint(0, len(audio_sets_yes))]}")

    elif ans == 2: # No
        print("No, delay : ", delay)
        if np.random.rand() < backchannel_chance: # chance of backchannel
                upload_audio_to_misty(misty, f"audios/{audio_sets_backchannel[np.random.randint(0, len(audio_sets_backchannel))]}")
                move_head_no(misty, 0)
                upload_audio_to_misty(misty, f"audios/{audio_sets_no[np.random.randint(0, len(audio_sets_no))]}")

    elif ans == 3: # Maybe
        print("Maybe, delay : ", delay)
        if np.random.rand() < backchannel_chance:
            upload_audio_to_misty(misty, f"audios/{audio_sets_backchannel[np.random.randint(0, len(audio_sets_backchannel))]}")
            move_head_backchanneling(misty, 0)
            upload_audio_to_misty(misty, f"audios/{audio_sets_maybe[np.random.randint(0, len(audio_sets_maybe))]}")


    return

def main_process():
    global guiData
    
    upload_audio_to_misty(misty, "audios/intro.wav")
    time.sleep(0.5)
    start_streaming(misty) #havent tested this yet
    while True:
        delay = guiData.get("delay_enabled")
        ans = guiData.get("ans")
        animal_num = guiData.get("animal_num")
        prompt = guiData.get("prompt")

        if delay and ans != 0:
            time.sleep(delay_duration)

        if ans != 0:
            handle_answer(ans, delay)
        elif prompt != 0:
            audio_handler.handle_prompt(misty, prompt)

        
        # reset answer
        guiData["ans"] = 0 
        guiData["animal_num"] = 0
        guiData["prompt"] = 0

        time.sleep(0.05)


if __name__ == "__main__":

    thread = threading.Thread(target=main_process)
    thread.start()
    app.run(debug=False)