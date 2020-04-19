import cv2
import numpy as np
import time

"""
ORIG SHAPE : 480 x 640
"""

WIDTH = 160
HEIGHT = 120


if __name__ == "__main__":
    cap = cv2.VideoCapture("data/train.mp4")
    speeds = list(map(float, open("data/train.txt", "r").read().split('\n')))

    X = []
    Y = []

    for i in range(2401):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        frame = frame/255.

        X.append(frame)
        Y.append(speeds[i])

        print(i)

        cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    np.savez_compressed("data/data.npz", X, Y)
