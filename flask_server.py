from flask import Flask, request, jsonify
import threading
import time
from misty_functions import move_head_no, move_head_yes, play_audio, upload_audio_to_misty, move_head_backchanneling, move_head_backchanneling2, start_face_detection, get_face_data
from mistyPy.Robot import Robot

app = Flask(__name__)

misty = Robot("192.168.1.237")

guiData = {"answer": "idle"}
old_data = guiData


@app.route("/")
def index():
    return open("index_alex.html").read()


@app.route("/process", methods=["POST"])
# def process():
#     global guiData
#     # Get data from the button press (example variable)
#     data = request.json
#     print("Received data:", data)

#     # Do something with the data, e.g., run a function
#     guiData = data

#     # Return a response
#     return jsonify({"result": True})

def process():
    data = request.json
    print("Received data:", data)  # Debugging
    global guiData
    guiData = data  # Assign received data to global variable
    return jsonify({"status": "success"})


def main_process():
    global old_data, guiData
    upload_audio_to_misty(misty, "audios/intro.wav")
    start_face_detection(misty)
    while True:
        faces = get_face_data(misty)
        print(faces)

        if guiData.get("answer") == "yes":
            upload_audio_to_misty(misty, "audios/yes.wav")
            move_head_yes(misty, 0)
            guiData = old_data
        elif guiData.get("answer") == "no":
            upload_audio_to_misty(misty, "audios/no.wav")
            move_head_no(misty, 0)
            guiData = old_data
            pass
        elif guiData.get("answer") == "nod yes":
            move_head_yes(misty, 0)
            guiData = old_data
            pass
        elif guiData.get("answer") == "nod no":
            move_head_no(misty, 0)
            guiData = old_data
            pass
        elif guiData.get("answer") == "yeah":
            upload_audio_to_misty(misty, "audios/yeah.wav")
            move_head_yes(misty, 0)
            guiData = old_data
            pass
        elif guiData.get("answer") == "right":
            upload_audio_to_misty(misty, "audios/right.wav")
            move_head_yes(misty, 0)
            guiData = old_data
            pass
        elif guiData.get("answer") == "nah!":
            upload_audio_to_misty(misty, "audios/nah.wav")
            move_head_no(misty, 0)
            guiData = old_data
            pass
        elif guiData.get("answer") == "try again":
            upload_audio_to_misty(misty, "audios/try_again.wav")
            move_head_no(misty, 0)
            guiData = old_data
            pass
        elif guiData.get("answer") == "mhm":
            upload_audio_to_misty(misty, "audios/mmhmm.wav")
            move_head_backchanneling(misty, 0)
            guiData = old_data
            pass
        elif guiData.get("answer") == "aha":
            upload_audio_to_misty(misty, "audios/ah_ha.wav")
            move_head_backchanneling(misty, 0)
            guiData = old_data
            pass
        elif guiData.get("answer") == "uh-huh":
            upload_audio_to_misty(misty, "audios/uh_huh.wav")
            move_head_backchanneling2(misty, 0)
            guiData = old_data
            pass
        elif guiData.get("answer") == "hmm":
            upload_audio_to_misty(misty, "audios/hmmmm.wav")
            move_head_backchanneling2(misty, 0)
            guiData = old_data
            pass
        time.sleep(0.1)


if __name__ == "__main__":
    thread = threading.Thread(target=main_process)
    thread.start()
    app.run(debug=False)
