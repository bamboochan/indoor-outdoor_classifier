'''
This script runs the trained module (in our case: the indoor/outdoor classifier) on a given image

the read_tensor_from_image_file funciton is adopted from
https://www.tensorflow.org/tutorials/images/image_recognition

To run:
make sure the graph final_output_graph.pb is in the same folder as this python file
'''


import os
import numpy as np
import tensorflow as tf
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


if __name__ == "__main__":

  print('Loading...')

  model_file = 'final_output_graph.pb'
  input_height = 299
  input_width = 299
  input_layer = 'Placeholder'
  output_layer = 'final_layer'

  graph = load_graph(model_file)

  while True:

    print("Input absolute path to image file, or input q to exit:")
    file_name = sys.stdin.readline()[0:-1]

    if file_name == "q":
      break

    _, extension = os.path.splitext(file_name)

    if not os.path.exists(file_name):
      print('ERROR: File does not exist')
      continue

    if extension.lower() not in ['.jpg', '.jpeg', '.bmp', '.gif', '.png']:
      print("ERROR: Format not supported. Supported formats: .jpg; .jpeg, .png, .bmp, .gif")
      continue

    t = read_tensor_from_image_file(
        file_name,
        input_height=input_height,
        input_width=input_width)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0], {
          input_operation.outputs[0]: t
      })
    results = np.squeeze(results)

    labels = ["indoors", "outdoors"]
    for i in range(2):
      print(labels[i], results[i])
    if results[0] > results[1]:
      print("Verdict: indoors\n")
    else:
      print("Verdict: outdoors\n")
