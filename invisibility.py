import cv2
import time


# initialize the HOG descriptor
fd = cv2.HOGDescriptor()
fd.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam
vs = cv2.VideoCapture(0, cv2.CAP_MSMF)

# 3 seconds in order to launch the program and clear the webcam view
time.sleep(3)
st_fr = 0

# first frame is too dark, taking 19 th frame as anchor point
while True:
    _, fr = vs.read()
    st_fr = st_fr + 1

    if st_fr == 20:
        fr = cv2.resize(fr, (640, 480))
        fr1 = fr.copy()
        break


# reinstantiate frame to iterate over the video
_, fr = vs.read()
#frame = cv2.rotate(frame,rotateCode = 1) 
fr = cv2.resize(fr, (640, 480))

while True:
    _, fr = vs.read()
    fr = cv2.resize(fr, (640, 480))
    bw = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
    boxes, weights = fd.detectMultiScale(bw,
                                          winStride=(4, 4),
                                          padding=(16, 16),
                                          scale=1.05)
    org = fr.copy()
    rect = fr.copy()
    
    for (x, y, w, h) in boxes:

        cropped = fr1[y:y + h, x:x + w]
        fr[y:y + h, x:x + w] = cropped
        cv2.rectangle(rect, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # also for debugging
    cv2.imshow('Current Frame', fr)
    cv2.imshow('Rectangle Frame', rect)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

vs.release()
cv2.destroyAllWindows()
cv2.waitKey(1)