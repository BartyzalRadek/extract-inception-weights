# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Simple extraction of specific layer weights from Inception.

Use optional arguments:

--model_dir = Path to classify_image_graph_def.pb. Default is 'model_dir'. 
              If it doesn't exist it will be created and Inception net will be automatically downloaded to it.
--image_dir = Path to directory containing images. Default is 'img'.
--embed_dir = Path to embedding dir - another directory will be created inside, named after the chosen layer. Default is 'embeddings'.
--layer = Name of the hidden layer to extract weights from. Default is 'pool_3:0' which is the next to last layer..

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import listdir
from os.path import isfile, join
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# pwd = os.getcwd()
pwd = '.'

tf.app.flags.DEFINE_string(
    'model_dir', os.path.join(pwd, 'model_dir'),
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_dir', os.path.join(pwd, 'img'),
                           """Path to directory containing images.""")
tf.app.flags.DEFINE_string(
    'embed_dir', os.path.join(pwd, 'embeddings'),
    """Path to embedding dir.""")

tf.app.flags.DEFINE_string(
    'layer', 'pool_3:0',
    """Name of the hidden layer to extract weights from..""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
"""# pylint: enable=line-too-long"""

DELIMITER = ' '

def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.
  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image_data, layer_name, sess):
    """Runs inference on an image.
    Args:
      image_data: Image data by tf.gfile.FastGFile(image, 'rb').read()
      layer_name: Name of the hidden layer. e.g.: 'pool_3:0'
      sess: The current active TensorFlow Session.
    Returns:
      Embedding values.
    """
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    # softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

    feature_tensor = sess.graph.get_tensor_by_name(layer_name)
    embedding_values = np.squeeze(sess.run(feature_tensor, {'DecodeJpeg/contents:0': image_data}))
    return embedding_values


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def get_JPEGs_from_dir(dir_name):
    return [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f)) and '.jpg' in f]

def create_embedding_file(image_path, sess, layer_name, i, total):
    embedding_path = os.path.join(
      FLAGS.embed_dir, layer_name.replace(":", "_"), image_path.split(os.sep)[-1] + '.txt')

    if not os.path.exists(embedding_path):
        print('Creating embedding file {}/{} {}'.format(i, total, embedding_path))
        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        embedding_values = run_inference_on_image(image_data, layer_name, sess)
        embedding_string = DELIMITER.join(str(x) for x in embedding_values)
        with open(embedding_path, 'w') as embedding_file:
            embedding_file.write(embedding_string)
    else:
        print('Embedding file {}/{} {} already exists.'.format(i, total, embedding_path))

def create_all_embeddings(images, layer_name):
    ensure_dir_exists(os.path.join(FLAGS.embed_dir, layer_name.replace(":", "_")))
    i = 1
    total = len(images)
    with tf.Session() as sess:
        for img in images:
            create_embedding_file(img, sess, layer_name, i, total)
            i += 1


def main(_):
    maybe_download_and_extract()
    # Creates graph from saved GraphDef.
    create_graph()
    images = get_JPEGs_from_dir(FLAGS.image_dir)

    create_all_embeddings(images, FLAGS.layer)


if __name__ == '__main__':
    tf.app.run()
