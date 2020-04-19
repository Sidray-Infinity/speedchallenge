import cv2
import time


if __name__ == "__main__":
    cap = cv2.VideoCapture("data/train.mp4")

    while cap.isOpened():
        ret, frame = cap.read()

        cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.release()
    cv2.destroyAllWindows()
