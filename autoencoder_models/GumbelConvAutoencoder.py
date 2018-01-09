import tensorflow as tf
import sys
from commons.ops import *

layers = tf.contrib.layers

class GumbelAutoencoder(object):

    def __init__(self, n_input, n_hidden, optimizer = tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden

        data_dim = 784

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.tau = tf.placeholder(tf.float32)

        _x = tf.reshape(self.x,[-1,28,28,1])
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
        self.hidden = z
        _x = z

        # decode _x and test _t
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
        x_hat = _x
        with tf.name_scope('output_prereshape'):
        tf.summary.image('output_prereshape', y, 2)
        tf.summary.image('output_prereshape_test', _t, 2)


        # sample from gaussian distribution
        '''
        eps = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.n_hidden]), 0, 1, dtype = tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.reconstruction = tf.add(tf.matmul(self.z, self.weights['w2']), self.weights['b2'])
        '''

        # cost
        self.cost = tf.losses.mean_squared_error(x,tf.reshape(x_hat,[-1,784]))
        #self.cost = gumbel_loss(self.x, self.x_hat)
        '''
        reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
        '''
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver()

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['log_sigma_w1'] = tf.get_variable("log_sigma_w1", shape=[self.n_input, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['log_sigma_b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X, tau):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.tau: tau})
        return cost

    def calc_total_cost(self, X, tau):
        return self.sess.run(self.cost, feed_dict = {self.x: X, self.tau: tau})

    def transform(self, X):
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.z: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

    def save(self, model_dir):
        return self.saver.save(self.sess, model_dir + '/model.ckpt')

    def restore(self, saved_model):
        return self.saver.restore(self.sess, saved_model)
