import train_network
import tensorflow as tf
import tensorflow_hub as hub

module_spec = hub.load_module_spec(train_network.module_url)

class TEST_create_image_lists(tf.test.TestCase):
  def test(self):
    result = train_network.create_image_lists(train_network.path_to_data + '/training_data/')
    self.assertEqual(list(result.keys()), ['indoors', 'outdoors'])
    self.assertGreater(len(result['indoors']), 0)
    self.assertGreater(len(result['outdoors']), 0)
    result = train_network.create_image_lists(train_network.path_to_data + '/test_data/')
    self.assertEqual(list(result.keys()), ['indoors', 'outdoors'])
    self.assertGreater(len(result['indoors']), 0)
    self.assertGreater(len(result['outdoors']), 0)


class TEST_create_module_graph(tf.test.TestCase):
  def test(self):
    graph, features_tensor, resized_image_tensor = train_network.create_module_graph(module_spec)
    self.assertEqual(resized_image_tensor.shape.as_list(), [None, 299, 299, 3])

class TEST_add_jpeg_decoding(tf.test.TestCase):
  def test(self):
    jpeg_data, resized_image = train_network.add_jpeg_decoding(module_spec)
    self.assertEqual(resized_image.shape.as_list(), [1, 299, 299, 3])


if __name__ == '__main__':
  tf.test.main()



