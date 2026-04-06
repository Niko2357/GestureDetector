import cv2
import mediapipe as mp
import csv
import os


def log_gesture_data():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    file_name = 'gestures_dataset.csv'

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
        if key == ord('r'):
            label = 0
        elif key == ord('p'):
            label = 1
        elif key == ord('s'):
            label = 2
        elif key == ord('q'):
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
