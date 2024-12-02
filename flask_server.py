from flask import Flask, request, jsonify
import threading
import time
from misty_functions import move_head_no, move_head_yes, play_audio
from mistyPy.Robot import Robot
import numpy as np

DEBUG = True

app = Flask(__name__)

misty = Robot("192.168.1.237")

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

guiData = {
    "ans": 0, # 0: no answer, 1: yes, 2: no, 3: maybe
    "sc_enabled": False,  # Social Cues
    "delay_enabled": False,  # Delay
    "animal_num": 0,  # 0: no animal, 1: lion, 2: Butterfly, 3: Cow, 4: Frog, 5: Shark
}

audio_sets_yes = ["yes.wav", "yeah.wav", "uh_huh.wav", "right.wav"]
audio_sets_no = ["no.wav", "no2.wav", "nah.wav", "wrong.wav"]
audio_sets_maybe = []
audio_sets_backchannel = ["hmmmm.wav"]
audio_sets_animals = ["animal1.wav", "animal2.wav", "animal3.wav", "animal4.wav", "animal5.wav"]
delay_duration = 1  # seconds
backchannel_chance = 1 # 50% chance of backchannel


@app.route("/")
def index():
    return open("misty_lab2.html").read()


@app.route("/process", methods=["POST"])
def process():
    global guiData
    # Get data from the button press (example variable)
    data = request.json
    # print("Received data:", data)

    # Do something with the data, e.g., run a function
    guiData = data

    # Return a response
    return jsonify({"result": True})

def handle_answer(ans, social_cues, delay):
    if ans == 1: # Yes
        print("Yes, social cues : ", social_cues, "delay : ", delay)
        if social_cues:
            if np.random.rand() > backchannel_chance: # chance of backchannel
                    play_audio(misty, audio_sets_backchannel[np.random.randint(0, len(audio_sets_backchannel))])
        #     move_head_yes(misty, 0)
        # play_audio(misty, audio_sets_yes[np.random.randint(0, len(audio_sets_yes))])

    elif ans == 2: # No
        print("No, social cues : ", social_cues, "delay : ", delay)
        if social_cues:
            if np.random.rand() > backchannel_chance: # chance of backchannel
                    play_audio(misty, audio_sets_backchannel[np.random.randint(0, len(audio_sets_backchannel))])
        #     move_head_no(misty, 0)
        # play_audio(misty, audio_sets_no[np.random.randint(0, len(audio_sets_no))])

    elif ans == 3: # Maybe
        print("Maybe, social cues : ", social_cues, "delay : ", delay)
        if social_cues:
            if np.random.rand() > backchannel_chance:
                play_audio(misty, audio_sets_backchannel[np.random.randint(0, len(audio_sets_backchannel))])
            # move_head_maybe(misty, 0)
        # play_audio(misty, audio_sets_maybe[np.random.randint(0, len(audio_sets_maybe))])

    return

def main_process():
    global guiData
    while True:
        delay = guiData.get("delay_enabled")
        social_cues = guiData.get("sc_enabled")
        ans = guiData.get("ans")
        animal_num = guiData.get("animal_num")

        if delay and ans != 0:
            time.sleep(delay_duration)

        if ans != 0:
            handle_answer(ans, social_cues, delay)
        elif animal_num != 0:
            print("Animal number: ", animal_num) 
            # play_audio(misty, "false.wav") #misty will say "your guess is wrong,"
            # play_audio(misty, audio_sets_animals[animal_num-1]) # "the correct animal is {animal_name}"

        
        # reset answer
        guiData["ans"] = 0 
        guiData["animal_num"] = 0

        time.sleep(0.05)


if __name__ == "__main__":
    thread = threading.Thread(target=main_process)
    thread.start()
    app.run(debug=False)
