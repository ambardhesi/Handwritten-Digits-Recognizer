import numpy as np
import cv2

def load_data(dataset_path):
    dataset = np.genfromtxt(dataset_path, delimiter = ",", dtype = "uint8")
    target = np.array(dataset[:, 0], 'int')
    dataset = np.array(dataset[:, 1:].reshape(dataset.shape[0], 28, 28), 'int16')

    return (dataset, target)

def deskew(image):
    moments = cv2.moments(image)
    (h, w) = image.shape[:2]
    if abs(moments['mu02']) < 1e-2:
        return image.copy()
    skew = moments['mu11']/moments['mu02']
    M = np.float32([
        [1, skew, -0.5 * w * skew],
        [0, 1, 0]])
    image = cv2.warpAffine(image, M, (w, h), 
            flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
   
    return image
  

