# Indoors/Outdoors Image Classifier

This is an image classifier which, given a image, outputs the probability of it being shot indoor or outdoor.
It is trained on around 3000 images from each class from youtube 8m videos with relevant labels, with 100 training steps with 100 batch size.
Training data and testing data can be viewed in the training\_data or testing\_data folder in [my google drive link](https://drive.google.com/drive/folders/1A0W2QiL8Ezp6WLynyinQAAkcXymGVV8e?usp=sharing)
Please view report.txt for the details.

## Requirements
1. Python 3.6.1
2. OpenCV 3.4.3: For the extract\_frames or extract\_frames\_specific to select images from videos 
3. Tensorflow 1.11.0 
4. Tensorflow_hub 0.1.1
5. connection to the internet: for train_network to get the pretrained module from tensorflow

## Installation

### To use the classifier (Python)
1. clone the repository. Make sure the file final_output_graph.pb is in the same directory as classifer.py
3. make sure the requirements mentioned above are fulfilled
2. run 
```
  python3 classifier.py
```
  It will prompt you to enter a path to the image that you want to classify. Please don't add the escape character if a folder name has a space

### To use the classifier (Command line)
1. download the classifier folder in [my google drive link](https://drive.google.com/drive/folders/1A0W2QiL8Ezp6WLynyinQAAkcXymGVV8e?usp=sharing)
2. run
```
./classifier
```

It will prompt you to enter a path to the image that you want to classify. Please don't add the escape character if a folder name has a space.

### To retrain the model and test it on a custom set of images
1. clone the repository.
2. In the file train\_network, change path\_to\_data to a folder that contains two subfolders: training_data and testing_data
each having two subfolders indoors and outdoors, in which images are stored as jpeg
and change the save\_path to some path that the final graph will be saved to
3. run 
```
  python3 train_network.py 
```
