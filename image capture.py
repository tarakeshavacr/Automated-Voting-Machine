import numpy as np
import cv2 as cv
i=0
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv.imwrite(str(i)+'.jpg',frame)
    i=i+1
    
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        
        break
cap.release()
cv.destroyAllWindows()
