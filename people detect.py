import cv2

def inside(r, q):
    (rx, ry), (rw, rh) = r
    (qx, qy), (qw, qh) = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

cv2.namedWindow("people detection demo", 1)
storage = cv2.CreateMemStorage(0) 
capture = cv2.CaptureFromCAM(0)
cv2.SetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 600)
cv2.SetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 450)
while True:
    img = cv2.QueryFrame(capture)
    found = list(cv2.HOGDetectMultiScale(img, storage, win_stride=(8,8),
                                     padding=(32,32), scale=1.05, group_threshold=2))
    found_filtered = []
    for r in found:
        insidef = False
        for q in found:
            if inside(r, q):
                insidef = True
                break
        if not insidef:
            found_filtered.append(r)
    for r in found_filtered:
        (rx, ry), (rw, rh) = r
        tl = (rx + int(rw*0.1), ry + int(rh*0.07))
        br = (rx + int(rw*0.9), ry + int(rh*0.87))
        cv2.Rectangle(img, tl, br, (0, 255, 0), 3)

    cv2.ShowImage("people detection demo", img)
    if cv2.WaitKey(10) == ord('q'):
        break