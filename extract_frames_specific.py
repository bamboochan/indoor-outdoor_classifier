'''
 This script will get the frames of the video at the url provided
 The frames are extracted uniformly every "interval" seconds, starting at "starttime"
 and ending at "endtime" (both in minutes)
 To run: change the variables "interval", "starttime", "endtime" and "url" accordingly
 Required libraries:
  OpenCV, tested with version 3.4.3
  Pafy, tested with verson 0.5.4
  youtube_dl, tested with version 2018.10.29
'''

import cv2
import pafy
import random
import string
import warnings

foldername_tosave = './test_data/outdoors/'
starttime = 3.5
endtime = 12.5
interval = 5
startframe = int(starttime * 60 / interval)
endframe = int(endtime * 60 / interval)

# this is suppressing warnings from not verifying SSL certificates when connecting to youtube
warnings.filterwarnings("ignore")



url = 'https://www.youtube.com/watch?v=N6QIFqX7OSQ'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="webm")

cap = cv2.VideoCapture(play.url)
n = 1
ret = True
imgs = []
while (ret):
  ret, frame = cap.read()
  imgs.append(frame)

n = len(imgs)
for j in range(startframe, endframe + 1):
  cv2.imwrite(foldername_tosave + ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.jpg',
              imgs[(n // (endframe + 1)) * j])


cap.release()
