'''
This script filters out video in youtube 8m that contains at least one labels of interest and save their ids.
To run: change the path variable to the folder containing all yt8m files
 and change the target_labels to indoor_labels or outdoor_labels
 and change the filename_tosave
'''
import tensorflow as tf
import os
import numpy as np
from google.protobuf.descriptor import FieldDescriptor
from copy import copy
from protobuf_to_dict import protobuf_to_dict, TYPE_CALLABLE_MAP

type_callable_map = copy(TYPE_CALLABLE_MAP)
# convert TYPE_BYTES to a Python bytestring

type_callable_map[FieldDescriptor.TYPE_BYTES] = str
indoor_labels = [416, 514, 380, 307, 1245, 3501, 3566]
outdoor_labels = [2434, 666, 2211, 2105, 2230, 419]

path = "/Users/user/Documents/yt8m/"
target_labels = outdoor_labels
filename_tosave = "outdoor_ids.txt"

directory = os.fsencode(path)
ids = []
i = 0

for file in os.listdir(directory):
  filename = os.fsdecode(file)
  if i >= 1000:
    break
  try:
    for example in tf.python_io.tf_record_iterator(path + filename):
      result = tf.train.Example.FromString(example)
      dict = protobuf_to_dict(result, type_callable_map=type_callable_map)
      labels = dict['features']['feature']['labels']['int64_list']['value']
      flag = False
      for x in target_labels:# specify here what lebels of interest are
        if x in labels:
          flag = True
      if flag:
        ids.append(dict['features']['feature']['id']['bytes_list']['value'][0][2:6])
        i += 1
  except:
    pass

np.savetxt(filename_tosave, np.array(ids), fmt='%s', delimiter='\n')


