from flask import Flask, request, jsonify
import threading
import time
from misty_functions import MistyController
import numpy as np
from io import BytesIO
from PIL import Image
from collections import deque
import cv2
import websocket
from misty_gazing import GazeTracker

DEBUG = True

app = Flask(__name__)

html = "htmls/misty_finalv3.html"  # html file path
misty = MistyController()

guiData = {
    "ans": 0,  # 0: no answer, 1: yes, 2: no, 3: maybe
    "delay_enabled": False,  # Delay
    "prompt": 0,
    "gaze": 1,
}

audio_sets_yes = ["yes.wav", "yeah.wav", "uh_huh.wav", "right.wav"]
audio_sets_no = ["no.wav", "no2.wav", "nah.wav", "sorry.wav"]
audio_sets_maybe = ["maybe.wav"]
audio_sets_backchannel = ["hmmmm.wav"]
audio_welcome = "intro.wav"


delay_duration = 1  # seconds
backchannel_chance = 0.99  # 50% chance of backchannel


@app.route("/")
def index():
    return open(html).read()


@app.route("/process", methods=["POST"])
def process():
    global guiData
    data = request.json
    print("Received data:", data)
    guiData = data
    return jsonify({"result": True})


def handle_answer(ans, delay):
    if ans == 1:  # Yes
        print("Yes, delay : ", delay)
        if np.random.rand() < backchannel_chance:  # chance of backchannel
            misty.upload_audio(
                f"audios/{audio_sets_backchannel[np.random.randint(0, len(audio_sets_backchannel))]}"
            )
            misty.move_head_yes(0)
            misty.upload_audio(
                f"audios/{audio_sets_yes[np.random.randint(0, len(audio_sets_yes))]}"
            )

    elif ans == 2:  # No
        print("No, delay : ", delay)
        if np.random.rand() < backchannel_chance:  # chance of backchannel
            misty.upload_audio(
                f"audios/{audio_sets_backchannel[np.random.randint(0, len(audio_sets_backchannel))]}"
            )
            misty.move_head_no(0)
            misty.upload_audio(
                f"audios/{audio_sets_no[np.random.randint(0, len(audio_sets_no))]}"
            )

    elif ans == 3:  # Maybe
        print("Maybe, delay : ", delay)
        if np.random.rand() < backchannel_chance:
            misty.upload_audio(
                f"audios/{audio_sets_backchannel[np.random.randint(0, len(audio_sets_backchannel))]}"
            )
            misty.move_head_backchanneling(0)
            misty.upload_audio(
                f"audios/{audio_sets_maybe[np.random.randint(0, len(audio_sets_maybe))]}"
            )

    return


# WebSocket video frame receiver
frame_queue = deque(maxlen=30)
last_processed_frame = None


def on_message(ws, message):
    img = Image.open(BytesIO(message))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # If the queue is full, remove the oldest frame before adding a new one
    if len(frame_queue) >= frame_queue.maxlen:
        frame_queue.popleft()

    frame_queue.append(frame)


def on_error(ws, error):
    print(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")


def on_open(ws):
    print("WebSocket connection opened")


def start_websocket_stream():
    global ws
    while True:
        try:
            ws = websocket.WebSocketApp(
                "ws://192.168.1.237:5678",
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )

            # Add reconnection mechanism
            ws.run_forever()  # Attempt to reconnect every 5 seconds

        except Exception as e:
            print(f"WebSocket connection error: {e}")
            print("Attempting to reconnect...")
            time.sleep(5)  # Wait before trying to reconnect


def stop_websocket_stream():
    if "ws" in globals():
        ws.close()


def handle_gaze(gaze, mistygaze):
    if frame_queue:
        # Always take the latest frame
        frame = frame_queue[-1]

        # Resize the frame to a more manageable size before processing
        resized_frame = cv2.resize(frame, (600, 800))

        processed_frame = mistygaze.process_frame(resized_frame, IsTracking=(gaze == 2))

        metrics = mistygaze.get_engagement_metrics()

        # Create text for display
        engagement_text = f"Engagement: {metrics['engagement_percentage']:.1f}%"
        looking_text = f"Looking: {metrics['is_looking']}"
        look_count_text = f"Look Count: {metrics['look_count']}"

        # Display metrics on the frame
        cv2.putText(
            processed_frame,
            looking_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if metrics["is_looking"] else (0, 0, 255),
            2,
        )

        cv2.putText(
            processed_frame,
            engagement_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            processed_frame,
            look_count_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        # print(f"Engagement Percentage: {metrics['engagement_percentage']:.2f}%")
        # print(f"Average Look Duration: {metrics['average_look_duration']:.2f} frames")
        # print(f"Look Count: {metrics['look_count']}")

        # cv2.putText(processed_frame,
        #             f"Engagement: {metrics['engagement_percentage']:.1f}%",
        #             (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Processed Frame", processed_frame)
        cv2.waitKey(1)  # Always call this to update the window


def main_process():
    global guiData
    # upload_audio_to_misty(misty, "audios/intro.wav")
    time.sleep(0.5)
    mistygaze = GazeTracker()

    # Start WebSocket server in a separate thread with exception handling
    ws_thread = threading.Thread(target=start_websocket_stream, daemon=True)
    ws_thread.start()

    while True:
        delay = guiData.get("delay_enabled")
        ans = guiData.get("ans")
        prompt = guiData.get("prompt")
        gaze = guiData.get("gaze")

        if gaze != 1:  # 1 = no gaze, 2 = gaze
            print(gaze)
            handle_gaze(gaze, mistygaze)
        else:
            cv2.destroyAllWindows()

        if delay and ans != 0:
            time.sleep(delay_duration)

        if ans != 0:
            handle_answer(ans, delay)
        elif prompt != 0:
            misty.handle_audio_prompt(prompt)

        # reset answer
        guiData["ans"] = 0
        guiData["animal_num"] = 0
        guiData["prompt"] = 0
        guiData["gaze"] = 1

        time.sleep(0.05)


if __name__ == "__main__":

    thread = threading.Thread(target=main_process)
    thread.start()
    app.run(debug=False)
