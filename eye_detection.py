import numpy as np
import cv2

eye_cascade = cv2.CascadeClassifier('1/haarcascade_lefteye_2splits.xml')
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye = eye_cascade.detectMultiScale(gray, 1.3, 5)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    cv2.imwrite('eye.jpg',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
