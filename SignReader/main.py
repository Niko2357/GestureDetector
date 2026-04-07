import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import time
import sys
import mediapipe.python.solutions.hands as mp_hands


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def menu():
    img = np.zeros((480, 640, 3), np.uint8)
    img[:] = (25, 25, 25)

    cv2.rectangle(img, (0, 0), (640, 60), (45, 45, 45), -1)
    cv2.putText(img, "AI Gesture Detector", (140, 40), 0, 0.8, (255, 255, 255), 2)

    cv2.rectangle(img, (100, 100), (540, 380), (60, 60, 60), 2)

    cv2.circle(img, (180, 210), 10, (0, 255, 0), -1)
    cv2.putText(img, "1 to START", (210, 220), 0, 0.7, (200, 200, 200), 2)

    cv2.circle(img, (180, 280), 10, (0, 0, 255), -1)
    cv2.putText(img, "Escape to Exit", (210, 290), 0, 0.7, (200, 200, 200), 2)

    cv2.imshow('Menu', img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            cv2.destroyWindow('Menu')
            return True
        if key == 27:
            cv2.destroyWindow('Menu')
            return False


def run():
    models = {
        "RPS": {
            "file": "rps_model.pkl",
            "scaler": "scaler.pkl",
            "gestures": {0: "Rock", 1: "Paper", 2: "Scissors", 3: "switch"}
        },
        "Alphabet": {
            "file": "alphabet_model.pkl",
            "scaler": "alphabet_scaler.pkl",
            "gestures": {
                0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
                8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
                15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V",
                22: "W", 23: "Y", 24: "Z", 25: ".", 26: ",", 27: "?", 28: " ", 29: "switch"
            }
        },
        "Numbers": {
            "file": "num_model.pkl",
            "scaler": "num_scaler.pkl",
            "gestures": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "switch"}
        }
    }

    mode = "Alphabet"
    all_modes = list(models.keys())

    def load(name):
        setting = models[name]
        model_path = resource_path(setting["file"])
        scaler_path = resource_path(setting["scaler"])

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            m = joblib.load(model_path)
            s = joblib.load(scaler_path)
            return m, s, setting["gestures"]
        return None, None, None

    model, scaler, letter = load(mode)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if not ret:
        error_window = np.zeros((300, 500, 3), np.uint8)
        error_window[:] = (20, 20, 20)

        cv2.rectangle(error_window, (0, 0), (500, 50), (0, 0, 255), -1)
        cv2.putText(error_window, "Camera not found", (70, 35), 0, 0.7, (255, 255, 255), 2)

        cv2.putText(error_window, "1. Check if camera is plugged in.", (50, 120), 0, 0.6, (200, 200, 200), 1)
        cv2.putText(error_window, "2. Close other apps using camera.", (50, 160), 0, 0.6, (200, 200, 200), 1)
        cv2.putText(error_window, "3. Try a different USB port.", (50, 200), 0, 0.6, (200, 200, 200), 1)

        cv2.putText(error_window, "Press any key to return to Menu", (100, 260), 0, 0.5, (0, 255, 0), 1)

        cv2.imshow("Camera Error", error_window)
        cv2.waitKey(0)
        cv2.destroyWindow("Camera Error")
        cap.release()
        return

    sentence = ""
    last_detected = None
    start_time = None
    req_time = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        message = "Looking for hand..."

        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                data = []
                zero_x = hand.landmark[0].x
                zero_y = hand.landmark[0].y
                zero_z = hand.landmark[0].z

                for point in hand.landmark:
                    data.append(point.x - zero_x)
                    data.append(point.y - zero_y)
                    data.append(point.z - zero_z)

                if model is not None:
                    entry = np.array(data).reshape(1, -1)
                    entry_scaled = scaler.transform(entry)
                    num = model.predict(entry_scaled)[0]
                    message = letter.get(num, "?")

        if message not in ["Looking for hand...", "switch"]:
            if message == last_detected:
                elaps = time.time() - start_time
                cv2.rectangle(frame, (0, 60), (int((elaps / req_time) * w), 70), (0, 255, 255), -1)
                if elaps >= req_time:
                    sentence += message
                    last_detected = None
                    start_time = time.time()
            else:
                last_detected = message
                start_time = time.time()
        elif message == "switch":
            if last_detected == "switch":
                if time.time() - start_time >= 2.0:
                    mode = all_modes[(all_modes.index(mode) + 1) % len(all_modes)]
                    model, scaler, letter = load(mode)
                    last_detected = None
                    time.sleep(0.5)
            else:
                last_detected = "switch"
                start_time = time.time()
        else:
            last_detected = None

        cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.putText(frame, mode, (w - 170, 35), 0, 0.8, (255, 255, 255), 2)
        cv2.rectangle(frame, (0, 0), (w - 180, 60), (30, 30, 30), -1)

        cv2.putText(frame, "Detecting: " + message, (20, 40), 0, 0.8, (0, 255, 0), 2)

        cv2.rectangle(frame, (0, h - 80), (w, h), (50, 50, 50), -1)

        cursor = ""
        if int(time.time() * 2) % 2 == 0:
            cursor = "_"

        cv2.putText(frame, "Text: " + sentence + cursor, (20, h - 35), 0, 1, (255, 255, 255), 2)

        cv2.imshow('AI Write', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('m'):
            mode = all_modes[(all_modes.index(mode) + 1) % len(all_modes)]
            model, scaler, letter = load(mode)
            last_detected = None
        elif key == 13:
            if message != "Looking for hand..." and message != "?":
                sentence = sentence + message[0]
        elif key == ord(' '):
            sentence = sentence + " "
        elif key == 8:
            sentence = sentence[:-1]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    while True:
        if menu():
            run()
        else:
            break


# pyinstaller --noconfirm --onefile --windowed --name "GestureDetector" --additional-hooks-dir=hooks --collect-all
# mediapipe --collect-all sklearn --hidden-import sklearn.ensemble._forest --hidden-import sklearn.tree._utils
# --add-data "alphabet_model.pkl;." --add-data "num_model.pkl;." --add-data "num_scaler.pkl;." --add-data "alphabet_scaler.pkl;." --add-data "rps_model.pkl;." --add-data
# "scaler.pkl;." main.py
