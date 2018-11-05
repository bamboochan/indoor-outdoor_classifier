'''
This script will get the frames of videos whose ids are stored in filename
To run: change the variables filename, foldername_tosave, numframe and numvideo accordingly
    it will use numvideo number of videos from the filename file 
    and extract numframe number of frames from each video uniformly.
  Required libraries:
    OpenCV, tested with version 3.4.3
    Pafy, tested with verson 0.5.4
    youtube_dl, tested with version 2018.10.29

'''
import cv2
import pafy
import random
import string
import requests
import warnings

filename = 'outdoor_ids.txt'
foldername_tosave = './outdoors/'
numframe = 30
numvideo = 100

# this is suppressing warnings from not verifying SSL certificates when connecting to youtube
warnings.filterwarnings("ignore")

ids = open(filename, 'r')
i = 0
for line in ids.readlines():
  try:
    line = line[0:-1]
    webpage = 'https://data.yt8m.org/2/j/i/' + line[0:2] + '/' + line + '.js'
    output = requests.get(webpage, verify=False).text
    parsed = output.split(",")[1][1:-3]
    print(parsed)

    url = 'https://www.youtube.com/watch?v=' + parsed
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
    for j in range(1, numframe + 1):
      cv2.imwrite(foldername_tosave + ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.jpg',
                  imgs[(n // (numframe + 1)) * j])
    i += 1

    cap.release()
  except:
    pass

  print(i)
  if i >= numvideo:
    break
