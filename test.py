import numpy as np, cv2 as cv

cap = cv.VideoCapture('video.mp4')
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('outputvideo.avi', fourcc, 20.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        frame = cv.flip(frame,0)
        out.write(frame)
        
        cv.imshow('Video frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv.destroyAllWindows()