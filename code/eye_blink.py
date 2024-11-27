import numpy as np
import cv2
import time

# Initialize the face and eye cascade classifiers from XML files
face_cascade = cv2.CascadeClassifier(r'K:\Python\Computer Vision Detection Projects\Eye Blink Detection\Data\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'K:\Python\Computer Vision Detection Projects\Eye Blink Detection\Data\haarcascade_eye_tree_eyeglasses.xml')

# Video capture
cap = cv2.VideoCapture(0)
first_read = True

# Timer variables
start_time = None
elapsed_time = 0
paused_time = 0
timer_running = False

while True:
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_face = gray[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_face, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

            if len(eyes) >= 2:
                if first_read:
                    cv2.putText(img, "Eye detected, press 's' to begin", (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "Eyes open!", (70, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                # Resume timer
                if not timer_running:
                    if start_time is None:
                        start_time = time.time()
                    else:
                        start_time = time.time() - paused_time
                    timer_running = True
            else:
                if first_read:
                    cv2.putText(img, "No eyes detected", (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
                else:
                    # Pause timer and update elapsed time
                    if timer_running:
                        paused_time = time.time() - start_time
                        timer_running = False
                    print("Blink detected--------------")
                    cv2.waitKey(3000)
                    first_read = True

    else:
        cv2.putText(img, "No face detected", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    # Display timer
    if timer_running:
        current_time = time.time()
        elapsed_time = current_time - start_time
    else:
        elapsed_time = paused_time

    timer_text = f"Time: {elapsed_time:.2f} s"
    cv2.putText(img, timer_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.imshow('img', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if timer_running:
            # Stop the timer
            paused_time = time.time() - start_time
            timer_running = False
        else:
            # Restart the timer from zero
            start_time = time.time()
            paused_time = 0
            timer_running = True
            first_read = False

cap.release()
cv2.destroyAllWindows()
