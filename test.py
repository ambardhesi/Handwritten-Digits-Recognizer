from sklearn.externals import joblib
from HOG import HOG
import numpy as np
import dataset
import argparse
import cv2

''' Set up argument parser to get the previously stored model's path and the path of the image
which is to be tested'''

argparser = argparse.ArgumentParser()
argparser.add_argument("-m", "--model", required = True,
        help = "path to where the model will be stored")
argparser.add_argument("-i", "--image", required = True,
        help = "path to the image file")
args = vars(argparser.parse_args())

# Load the model from path
SVC_model = joblib.load(args["model"])

hog = HOG(orientations = 18, pixelsPerCell = (10, 10), cellsPerBlock = (1, 1), normalise = True)

# Load the image, convert it to grayscale, blur it, and obtain the edges
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)

# Find all the contours in the image
(_, contours, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find all the bounding rectangles of the contours
rects = [cv2.boundingRect(contour) for contour in contours]

# Loop over all the rectangles
for rect in rects:
    (x, y, w, h) = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 

    # Find the region of interest, and resize it
    roi = gray[y: y + h, x: x + w]
    roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
      
    # Threshold the region of interest
    thresh = roi.copy()
    _, thresh = cv2.threshold(thresh, 90, 255, cv2.THRESH_BINARY_INV)
    thresh = dataset.deskew(thresh)
  
    # Get the histogram and predict the number
    hist = hog.describe(thresh)  
    predict = SVC_model.predict(np.array([hist], 'float64'))
    cv2.putText(image, str(predict), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) , 2)

# Display the result
cv2.imshow("Image", image)
cv2.waitKey(0)




