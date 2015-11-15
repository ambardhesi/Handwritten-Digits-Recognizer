from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from HOG import HOG
import dataset
import argparse

''' Set up the argument parser which will get the CSV file and location where model 
is to be stored'''

argparser = argparse.ArgumentParser()
argparser.add_argument("-d", "--dataset", required = True,
        help = "path to the dataset file")
argparser.add_argument("-m", "--model", required = True,
        help = "path to where the model will be stored")
args = vars(argparser.parse_args())

(digits, labels) = dataset.load_data(args["dataset"])

hog = HOG(orientations = 18, pixelsPerCell = (10, 10), cellsPerBlock = (1, 1), normalise = True)

data = []
# Add histogram for each digit in a list 
for digit in digits:
    digit = dataset.deskew(digit) 
    hist = hog.describe(digit.reshape((28,28)))
    data.append(hist)

# Set up and train the model
SVC_model = LinearSVC()
SVC_model.fit(data, labels)

# Save the model to file
joblib.dump(SVC_model, args["model"], compress = 3)


