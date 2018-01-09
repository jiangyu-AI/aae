# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
based on mnist_with_summaries.py Jan 6 2018
https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
A simple MNIST classifier which displays summaries in TensorBoard.

This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

class Conv2d(object) :
    def __init__(self,name,input_dim,output_dim,k_h=4,k_w=4,d_h=2,d_w=2, # stride = 2
                 stddev=0.02, data_format='NCHW') :
        with tf.variable_scope(name) :
            assert(data_format == 'NCHW' or data_format == 'NHWC')
            self.w = tf.get_variable('w', [k_h, k_w, input_dim, output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim], initializer=tf.constant_initializer(0.0))
            if( data_format == 'NCHW' ) :
                self.strides = [1, 1, d_h, d_w]
            else :
                self.strides = [1, d_h, d_w, 1]
            self.data_format = data_format
    def __call__(self,input_var,name=None,w=None,b=None,**kwargs) :
        w = w if w is not None else self.w
        b = b if b is not None else self.b

        if( self.data_format =='NCHW' ) :
            return tf.nn.bias_add(
                        tf.nn.conv2d(input_var, w,
                                    use_cudnn_on_gpu=True,data_format='NCHW',
                                    strides=self.strides, padding='SAME'),
                        b,data_format='NCHW',name=name)
        else :
            return tf.nn.bias_add(
                        tf.nn.conv2d(input_var, w,data_format='NHWC',
                                    strides=self.strides, padding='SAME'),
                        b,data_format='NHWC',name=name)
    def get_variables(self):
        return {'w':self.w,'b':self.b}

class TransposedConv2d(object):
    def __init__(self,name,input_dim,out_dim,
                 k_h=4,k_w=4,d_h=2,d_w=2,stddev=0.02,data_format='NCHW') :
        with tf.variable_scope(name) :
            self.w = tf.get_variable('w', [k_h, k_w, out_dim, input_dim],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[out_dim], initializer=tf.constant_initializer(0.0))

        self.data_format = data_format
        if( data_format =='NCHW' ):
            self.strides = [1, 1, d_h, d_w]
        else:
            self.strides = [1, d_h, d_w, 1]

    def __call__(self,input_var,name=None,**xargs):
        shapes = tf.shape(input_var)
        if( self.data_format == 'NCHW' ):
            shapes = tf.stack([shapes[0],tf.shape(self.b)[0],shapes[2]*2,shapes[3]*2])
        else:
            shapes = tf.stack([shapes[0],shapes[1]*2,shapes[2]*2,tf.shape(self.b)[0]])

        return tf.nn.bias_add(
            tf.nn.conv2d_transpose(input_var,self.w,output_shape=shapes,
                                data_format=self.data_format,
                                strides=self.strides,padding='SAME'),
            self.b,data_format=self.data_format,name=name)


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    fake_data=FLAGS.fake_data)

  sess = tf.Session()
  #sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.int64, [None], name='y-input')
    tau = tf.placeholder(tf.float32, name='tau')
    tf.summary.scalar('tau', tau)

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 2)

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

  _x = image_shaped_input # _*28*28*1
# encode
  enc1 = Conv2d('enc1',1,16,data_format='NHWC')
  _x = tf.nn.relu(enc1(_x)) # _*14*14*16
  enc2 = Conv2d('enc2',16,32,data_format='NHWC')
  _x = tf.nn.relu(enc2(_x)) # _*7*7*32 

  # gumbel softmax
  logits_y  = _x
  def sampling(logits_y):
      U = tf.random_uniform(tf.shape(logits_y), 0, 1)
      y = logits_y - tf.log(-tf.log(U + 1e-20) + 1e-20) # logits + gumbel noise
      y = tf.nn.softmax(y / tau)
      #y = tf.nn.softmax(tf.reshape(y, (-1, N, M)) / self.tau)
      #y = tf.reshape(y, (-1, N*M))
      return y
  z = sampling(logits_y)
  tf.summary.histogram('sampling gumbel softmax',z)
  tf.summary.tensor_summary('z',z)
  hidden = z
  #hidden = _x
  _x = z

  # for testing
  D = 32
  argmax_logits = tf.argmax(logits_y, axis=-1)
  latent_index = argmax_logits
  _t = tf.one_hot(argmax_logits, D,axis=-1)

  dec1 = TransposedConv2d('dec1',32,16,data_format='NHWC')
  _x = tf.nn.relu(dec1(_x)) #_*14*14*16
  _t = tf.nn.relu(dec1(_t))
  dec2 = TransposedConv2d('dec2',16,1,data_format='NHWC')
  _x = tf.nn.relu(dec2(_x)) #_*14*14*16
  _t = tf.nn.relu(dec2(_t))
  _x = tf.tanh(_x) # _*28*28*1 map values into range (-1,1)
  _t = tf.tanh(_t)
  y = _x
  with tf.name_scope('output_prereshape'):
    tf.summary.image('output_prereshape', y, 2)
    tf.summary.image('output_prereshape_test', _t, 2)

  x_hat = _x
  

  '''
  def gumbel_loss(x, x_hat):
      q_y = tf.reshape(logits_y, (-1, N, M))
      q_y = tf.nn.softmax(q_y)
      log_q_y = tf.log(q_y + 1e-20)
      kl_tmp = q_y * (log_q_y - tf.log(1.0/M))
      KL = tf.reduce_sum(kl_tmp, axis=(1, 2))
      elbo = tf.reduce_sum(data_dim * tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_hat))
      elbo = elbo - KL 
      return tf.reduce_sum(elbo)
  '''

  data_dim=784
  D = 32
  def gumbel_loss(x, x_hat):
      q_y = logits_y
      #q_y = tf.reshape(logits_y, (-1, N, M))
      q_y = tf.nn.softmax(q_y)
      log_q_y = tf.log(q_y + 1e-20)
      kl_tmp = q_y * (log_q_y - tf.log(1.0/(D)))
      KL = tf.reduce_sum(kl_tmp, axis=(1,2,3))
      #KL = tf.reduce_sum(kl_tmp, axis=(1, 2))
      elbo = tf.reduce_sum(data_dim * tf.nn.sigmoid_cross_entropy_with_logits(labels=image_shaped_input, logits=x_hat))
      elbo = elbo - KL 
      return tf.reduce_sum(elbo)

  #argmax_y = tf.reduce_max(tf.reshape(logits_y, (-1, N, M)), axis=-1, keep_dims=True)
  #self.argmax_y = tf.equal(tf.reshape(logits_y, (-1, N, M)), argmax_y)


  '''
  net = layers.fully_connected(net, 7 * 7 * 128)
  net = tf.reshape(net, [-1, 7, 7, 128])
  net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
  net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
  # Make sure that generator output is in the same range as `inputs`
  # ie [-1, 1].
  net = layers.conv2d(
    net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

  '''
  '''
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)
  '''

  # Do not apply softmax activation yet, see below.
  #y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)
  #y = nn_layer(hidden, 500, 784, 'layer2')
  with tf.name_scope('output_reshape'):
    image_shaped_output = tf.reshape(y, [-1, 28, 28, 1])
    tf.summary.image('output', image_shaped_output, 2)

  #with tf.name_scope('cross_entropy'):
  with tf.name_scope('loss'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.losses.sparse_softmax_cross_entropy on the
    # raw logit outputs of the nn_layer above, and then average across
    # the batch.
    with tf.name_scope('total'):
      #cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      #    labels=y_, logits=y)
      loss = tf.losses.mean_squared_error(x,tf.reshape(y,[-1,784]))
      #loss = gumbel_loss(image_shaped_input, x_hat)
  #tf.summary.scalar('cross_entropy', cross_entropy)
  tf.summary.scalar('loss', loss)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        loss)
        #cross_entropy)

  '''
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  '''

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  sess.run(tf.global_variables_initializer())

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train, tau_step):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    #return {x: xs, y_: ys, keep_prob: k}
    return {x: xs, y_: ys, tau:tau_step}

  anneal_rate = 0.0003
  min_temperature = 0.3
  test_temperature = 0.01
  tau_step = 5.0#, name="temperature")
  for i in range(FLAGS.max_steps):
    tau_step = np.max([tau_step * np.exp(- anneal_rate * i/500), min_temperature])
    if i % 100 == 0:  # Record summaries and test-set accuracy
      summary, loss_ = sess.run([merged, loss], feed_dict=feed_dict(False,test_temperature))
      #summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Loss at step %s: %s' % (i, loss_))
    else:  # Record train set summaries, and train
      if i % 1000 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True, tau_step),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True,tau_step))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()

  # extract latent index
  BATCH_SIZE = 100
  not_used = 0.01
  xs,ys = mnist.train.images, mnist.train.labels
  ys_notused = mnist.train.labels[0:BATCH_SIZE]
  ks = []
  for i in range(0,len(xs),BATCH_SIZE):
      batch = xs[i:i+BATCH_SIZE]
      k = sess.run(latent_index,feed_dict={x: batch, y_: ys_notused, tau:not_used})
      ks.append(k)
  ks = np.concatenate(ks,axis=0)
  np.savez(os.path.join(os.path.dirname(FLAGS.log_dir),'ks_ys.npz'),ks=ks,ys=ys)

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=10000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir',
                      type=str,
                      #default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                      #                     'tensorflow/mnist/input_data'),
                      default='/scratch/jy1367/datasets/mnist')
  parser.add_argument('--log_dir',
                      type=str,
                      #default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                      #                     'tensorflow/mnist/logs/mnist_with_summaries'),
                      default='log')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
