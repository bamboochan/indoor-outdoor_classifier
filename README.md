# Indoors/Outdoors Image Classifier

This is an image classifier which, given a image, outputs the probability of it being shot indoor or outdoor.
It is trained on around 3000 images from each class from youtube 8m videos with relevant labels, with 100 training steps with 100 batch size.

## Requirements
Python 3.6.1
OpenCV 3.4.3: For the extract_frames or extract_frames_specific to select images from videos 
Tensorflow 1.11.0 
Tensorflow_hub 0.1.1
connection to the internet: for train_network to get the pretrained module from tensorflow

## Installation

### To use the classifier (Python)
1. clone the repository. Make sure the file final_output_graph.pb is in the same directory as classifer.py
2. run 
  '''shell
  python3 classifier.py
  '''
  It will prompt you to enter a path to the image that you want to classify.


### To retrain the model and test it on a custom set of images
1. clone the repository.
2. In the file train_network, change path_to_data to a folder that contains two subfolders: training_data and testing_data
each having two subfolders indoors and outdoors, in which images are stored as jpeg
and change the save_path to some path that the final graph will be saved to
3. run 
  '''shell
  python train_network.py 
  '''
  
