# Handwritten-Digits-Recognizer
## Usage :
* Clone the repository
```
cd
git clone https://github.com/ambardhesi/Handwritten-Digits-Recognizer.git
```

* Download the training csv file from 
(https://www.kaggle.com/c/digit-recognizer/download/train.csv)
and place it in the same directory as train.py .

* Now train the classifer using the following command : 
```
python train.py --dataset train.csv --model dataset.pkl
 ```
This may take some time.

* Then test the classifier using the following command : 
```
python test.py --model dataset.pkl --image testimage.jpg
```
Replace testimage.jpg by your own image's name.
