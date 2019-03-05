# USAGE
# python image_diff.py --first images/original_01.png --second images/modified_01.png
# import the necessary packages
from skimage.measure import compare_ssim
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import imutils
import cv2
import math

# load the two input images
imageB = cv2.imread("./frames/frame1.jpg", cv2.IMREAD_COLOR)
imageA = cv2.imread("./frames/frame116.jpg", cv2.IMREAD_COLOR)


# convert the images to gray scale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    (x, y, w, h) = cv2.boundingRect(c)
    if w*h > 180:
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(imageA, ((x+(w//2)), (y + (h//2))), 2, (0, 0, 255), 1)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(imageB, ((x + (w//2)), (y + (h//2))), 2, (0, 0, 255), 1)
        location = [x + (w//2), y + (h//2)]
    print("Area = ", w*h)

# mark center of frame
(h, w) = imageA.shape[:2]
cv2.circle(imageA, (w//2, h//2), 7, (255, 255, 255), -1)

# find distance from center
center = [w//2, h//2]
distance = math.sqrt(((center[0]-location[0])**2)+((center[1]-location[1])**2))
distance = float("{0:.2f}".format(distance))
print("Distance = ", distance)

score = 0
if distance < 41:
    score = 10
elif distance < 81:
    score = 9
elif distance < 121:
    score = 8
elif distance < 161:
    score = 7
elif distance < 201:
    score = 6
elif distance < 241:
    score = 5
elif distance < 281:
    score = 4
elif distance < 321:
    score = 3
elif distance < 361:
    score = 2
else:
    score = 1
print("Score = " + str(score))
# draw line
lineThickness = 2
cv2.line(imageA, (center[0], center[1]), (location[0], location[1]), (0, 0, 255), lineThickness)

# draw text
dist = str(distance)
s = str(score)
cv2.putText(imageA, "Off by = " + dist + " Score = " + s, (location[0]+10, location[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255),
            lineType=cv2.LINE_AA)

# show the output images
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

