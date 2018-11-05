'''

This script will train a network from the features that we get from the pretrained module
to a final fully-connected layer with two neurons that will distinguish indoors and outdoors
and then this script will test the average accuracy of the network on the test data

This script is tested with:
tensorflow version: 1.11.0
tensroflow_hub version: 0.1.1 

Some parts of this file is adopted from:
https://www.tensorflow.org/hub/tutorials/image_retraining

To run: change path_to_data to a folder that contains two subfolders: training_data and testing_data
each having two subfolders indoors and outdoors, in which images are stored as jpeg
and change the save_path to some path that the final graph will be saved to
'''

import collections
from datetime import datetime
import os.path
import unittest
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


module_url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
path_to_data = '/Users/user/PycharmProjects/pex'
save_path = '/Users/user/PycharmProjects/pex/final_output_graph.pb'

def create_image_lists(path):
  '''
  Args:
    path: the folder that contains two subfolders, indoor and outdoor
  Returns:
    result: a dictionary that maps 'indoors' and 'outdoors' each to a list
    that contains all the image names in the indoors or outdoors subfolder
  '''
  result = collections.OrderedDict()


  for label in ['indoors', 'outdoors']:
    file_glob = path + label + "/*.jpg"
    file_list = tf.gfile.Glob(file_glob)
    training_images = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)

      training_images.append(base_name)
    result[label] = training_images
  return result



def create_module_graph(module_spec):

  ''' Creates a graph and loads Hub Module into it.

  Args:
    module_spec: the hub.ModuleSpec for the image module being used.

  Returns:
    graph: the tf.Graph that was created.
    features_tensor: the feature values tensor.
    resized_input_tensor: the input images tensor.
  '''

  height, width = hub.get_expected_image_size(module_spec)
  with tf.Graph().as_default() as graph:
    resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
    m = hub.Module(module_spec)
    features_tensor = m(resized_input_tensor)
  return graph, features_tensor, resized_input_tensor


def add_final_retrain_ops(class_count, features_tensor):
  '''Adds a new softmax and fully-connected layer for training and eval.

    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.

    The set up for the softmax and fully-connected layers is based on:
    https://www.tensorflow.org/tutorials/mnist/beginners/index.html

    Args:
      class_count: Integer of how many categories of things we're trying to
          recognize. (In our case it will be 2: indoor class and outdoor class)
      features_tensor: The output of the main CNN graph. i.e. the features

    Returns:
      The tensors for the training and cross entropy results, and tensors for the
      features input and ground truth input.
  '''
  batch_size, features_tensor_size = features_tensor.get_shape().as_list()

  with tf.name_scope('input'):
    features_input = tf.placeholder_with_default(
        features_tensor,
        shape=[batch_size, features_tensor_size],
        name='FeaturesInputPlaceholder')

    ground_truth_input = tf.placeholder(
        tf.int64, [batch_size], name='GroundTruthInput')

  layer_name = 'final_retrain_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal(
          [features_tensor_size, class_count], stddev=0.001)
      layer_weights = tf.Variable(initial_value, name='final_weights')

    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')

    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(features_input, layer_weights) + layer_biases

  final_tensor = tf.nn.softmax(logits, name='final_layer')


  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
        labels=ground_truth_input, logits=logits)

  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(0.001) 
    #I changed the loss function from Gradient descent to Adam for better accuracy
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, features_input, ground_truth_input,
          final_tensor)


def get_feature_from_image(path, sess, decoded_image_tensor, jpeg_data_tensor, resized_image_tensor, features_tensor):
  '''get the feature array for a given image
  Args:
    path: the path to the image
    sess: the tensorflow graph session
    decoded_image_tensor: the output tensor of the small graph which will resize the image\
      to make it the right size for the main CNN
    jpeg_data_tensor: the input tendor of the small graph that resize the image
    resized_image_tensor: the input tensor of the main CNN
    features_tensor: the tensor in the main CNN (second to last layer) that has feature
  Return:
    features: the output of the give image under after running through the resizing
      and the main CNN up to the feature layer
  '''
  image_binary = open(path, 'rb').read()
  resized_image = sess.run(decoded_image_tensor,
                                {jpeg_data_tensor: image_binary})

  features = sess.run(features_tensor,
                             {resized_image_tensor: resized_image})
  features = np.squeeze(features)

  return features


def get_random_features(features_dict, num):
  '''select num many element randomly from the values of the features_dict
  Args:
    features_dict: the dictionary mapping indoor/outdoor to a list of features
    num: the number of features to select
  Return:
    features_list: the list of length num that contains features of num images
    ground_truths: the list of length num that contains the gt of the features
      in features_list (0 for indoors and 1 for outdoors)
  '''
  features_list = []
  ground_truths = []
  num_of_indoors = len(features_dict['indoors'])
  num_of_outdoors = len(features_dict['outdoors'])

  for j in range(num):


    ind = np.random.randint(num_of_indoors + num_of_outdoors)
    if ind < num_of_indoors:
      label = 'indoors'
      answer = 0
    else:
      ind -= num_of_indoors
      label = 'outdoors'
      answer = 1

    features = features_dict[label][ind]
    features_list.append(features)
    ground_truths.append(answer)

  return features_list, ground_truths


def compute_all_features(lists, sess, test, decoded_image_tensor, jpeg_data_tensor, resized_image_tensor, features_tensor):
  ''' compute the features of all images in list and save the features
  Args:
    lists: a dictionary whose values are lists of filenames that correspond to images
    sess: the tensorflow graph session
    test: a bool that is true if we are testing. Then the path folder will be prepended test_
      and the saving path will be in the test_data folder as well
     decoded_image_tensor: the output tensor of the small graph which will resize the image\
      to make it the right size for the main CNN
    jpeg_data_tensor: the input tendor of the small graph that resize the image
    resized_image_tensor: the input tensor of the main CNN
    features_tensor: the tensor in the main CNN (second to last layer) that has feature
  Return;
    result: a dictionary with keys indoors and outdoors and values are computed features
  '''
  result = collections.OrderedDict()
  result['indoors'] = []
  result['outdoors'] = []
  if test:
    add = 'test_'
  else:
    add = 'training_'
  for i, filename in enumerate(lists['indoors']):
    if os.path.exists(path_to_data + '/' + add + 'features/indoors/' + filename.split('.')[0] + '.txt'):
      features = np.loadtxt(path_to_data + '/' + add + 'features/indoors/' + filename.split('.')[0] + '.txt', dtype=float, delimiter=',')
    else:
      path = path_to_data + '/' + add + 'data/indoors/' + filename
      features = get_feature_from_image(path, sess, decoded_image_tensor, jpeg_data_tensor, resized_image_tensor, features_tensor)
      np.savetxt(path_to_data + '/' + add + 'features/indoors/' + filename.split('.')[0] + '.txt', np.array(features), delimiter=',')
    result['indoors'].append(features)

    if i % 100 == 0:
      print(('Computed {} indoor ' + add + 'features').format(i))

  for i, filename in enumerate(lists['outdoors']):
    if os.path.exists(path_to_data + '/' + add + 'features/outdoors/' + filename.split('.')[0] + '.txt'):
      features = np.loadtxt(path_to_data + '/' + add + 'features/outdoors/' + filename.split('.')[0] + '.txt', dtype=float, delimiter=',')
    else:
      path = path_to_data + '/' + add + 'data/outdoors/' + filename
      features = get_feature_from_image(path, sess, decoded_image_tensor, jpeg_data_tensor, resized_image_tensor, features_tensor)
      np.savetxt(path_to_data + '/' + add + 'features/outdoors/' + filename.split('.')[0] + '.txt', np.array(features), delimiter=',')
    result['outdoors'].append(features)

    if i % 100 == 0:
      print(('Computed {} outdoor ' + add + 'features').format(i))
  return result





def add_jpeg_decoding(module_spec):
  '''Adds operations that perform JPEG decoding and resizing to the graph.
  Args:
    module_spec: The hub.ModuleSpec for the image module being used.

  Returns:
    Tensors for the node to feed JPEG data into, and the output of the
     preprocessing steps.
  '''
  input_height, input_width = hub.get_expected_image_size(module_spec)
  input_depth = hub.get_num_image_channels(module_spec)
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  # Convert from full range of uint8 to range [0,1] of float32.
  decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  return jpeg_data, resized_image

def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)

  # Create lists of all the images.
  image_lists = create_image_lists(path_to_data + '/training_data/')
  test_lists = create_image_lists(path_to_data + '/test_data/')

  class_count = 2

  # Set up the pre-trained graph.
  module_spec = hub.load_module_spec(module_url)
  graph, features_tensor, resized_image_tensor = (
      create_module_graph(module_spec))

  # Add the new layer that we'll be training.
  with graph.as_default():
    (train_step, cross_entropy, features_input,
     ground_truth_input, final_tensor) = add_final_retrain_ops(class_count, features_tensor)

  with tf.Session(graph=graph) as sess:
    # Initialize all weights: for the module to their pretrained values,
    # and for the newly added retraining layer to random initial values.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Set up the image decoding sub-graph.
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

    #compute/load(if cached already) all the testing/training features
    features_dict = compute_all_features(image_lists, sess, False, decoded_image_tensor, jpeg_data_tensor, resized_image_tensor, features_tensor)
    test_dict = compute_all_features(test_lists, sess, True, decoded_image_tensor, jpeg_data_tensor, resized_image_tensor, features_tensor)
   
    # Create the operations we need to evaluate the accuracy of our new layer.
    prediction = tf.argmax(final_tensor, 1)
    correct_prediction = tf.equal(prediction, ground_truth_input)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Run the training for as many cycles as we want
    num_of_training_steps = 100
    batch_size = 100

    for i in range(num_of_training_steps):
      # Get a batch of input feature values    
      train_features, train_ground_truth = get_random_features(features_dict, batch_size)
      # Feed the features and ground truth into the graph, and run a trainings step
      sess.run(
          train_step,
          feed_dict={features_input: train_features,
                     ground_truth_input: train_ground_truth})


      # Every so often, print out how well the graph is training.
      is_last_step = (i + 1 == num_of_training_steps)
      if (i % 100) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run(
            [evaluation_step, cross_entropy],
            feed_dict={features_input: train_features,
                       ground_truth_input: train_ground_truth})
        tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                        (datetime.now(), i, train_accuracy * 100))
        tf.logging.info('%s: Step %d: Cross entropy = %f' %
                        (datetime.now(), i, cross_entropy_value))

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.

    test_features = test_dict['indoors'] + test_dict['outdoors']
    test_ground_truth = [0]*len(test_dict['indoors']) + [1]*len(test_dict['outdoors'])

    test_accuracy, test_predictions = sess.run(
        [evaluation_step, prediction],
        feed_dict={
            features_input: test_features,
            ground_truth_input: test_ground_truth
        })
    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                    (test_accuracy * 100, len(test_features)))

    # Write out the trained graph and labels with the weights stored as
    # constants.
    tf.logging.info('Save final result to : ' + save_path)

    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['final_layer'])

    with tf.gfile.FastGFile(save_path, 'wb') as f:
      f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
  tf.app.run(main=main)
