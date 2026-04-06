import cv2
import mediapipe as mp
import csv
import os


def log_gesture_data():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    file_name = 'alphabet_dataset.csv'

    header = []
    for i in range(21):
        header.extend([f'x{i}', f'y{i}', f'z{i}'])
    header.append('target')

    if not os.path.exists(file_name):
        with open(file_name, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    print("Stiskni: 'r' (Rock), 'p' (Paper), 's' (Scissors), 'q' (Quit)")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        label = -1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            label = 0
        elif key == ord('b'):
            label = 1
        elif key == ord('c'):
            label = 2
        elif key == ord('d'):
            label = 3
        elif key == ord('e'):
            label = 4
        elif key == ord('f'):
            label = 5
        elif key == ord('g'):
            label = 6
        elif key == ord('h'):
            label = 7
        elif key == ord('i'):
            label = 8
        elif key == ord('j'):
            label = 9
        elif key == ord('k'):
            label = 10
        elif key == ord('l'):
            label = 11
        elif key == ord('m'):
            label = 12
        elif key == ord('n'):
            label = 13
        elif key == ord('o'):
            label = 14
        elif key == ord('p'):
            label = 15
        elif key == ord('q'):
            label = 16
        elif key == ord('r'):
            label = 17
        elif key == ord('s'):
            label = 18
        elif key == ord('t'):
            label = 19
        elif key == ord('u'):
            label = 20
        elif key == ord('v'):
            label = 21
        elif key == ord('w'):
            label = 22
        elif key == ord('y'):
            label = 23
        elif key == ord('z'):
            label = 24
        elif key == ord('Q'):
            break

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if label != -1:
                    row = []
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y
                    base_z = hand_landmarks.landmark[0].z

                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])

                    row.append(label)
                    with open(file_name, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    print(f"Uloženo gesto: {label}")

        cv2.imshow('Data Logger - Sběr dat pro AI', image)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    log_gesture_data()
