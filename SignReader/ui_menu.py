import cv2
import numpy as np

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

