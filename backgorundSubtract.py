import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while cap.isOpened():
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    frame = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
    out.write(frame)

    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

vidcap = cv2.VideoCapture('output.avi')
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite("./frames/frame%d.jpg" % count, image)     # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1