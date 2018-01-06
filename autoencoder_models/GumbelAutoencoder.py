import tensorflow as tf
import sys

class GumbelAutoencoder(object):

    def __init__(self, n_input, n_hidden, optimizer = tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model

        N = 30
        M = 10 
        data_dim = 784
        self.fc1 = tf.layers.Dense(512, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(256, activation=tf.nn.relu)
        self.fc3 = tf.layers.Dense(M*N)
        self.fc4 = tf.layers.Dense(256, activation=tf.nn.relu)
        self.fc5 = tf.layers.Dense(512, activation=tf.nn.relu)
        self.fc6 = tf.layers.Dense(784)

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.tau = tf.placeholder(tf.float32)

        logits_y  = self.fc3(self.fc2(self.fc1(self.x)))
        def sampling(logits_y):
            U = tf.random_uniform(tf.shape(logits_y), 0, 1)
            y = logits_y - tf.log(-tf.log(U + 1e-20) + 1e-20) # logits + gumbel noise
            y = tf.nn.softmax(tf.reshape(y, (-1, N, M)) / self.tau)
            y = tf.reshape(y, (-1, N*M))
            return y
        z = sampling(logits_y)
        self.x_hat = self.fc6(self.fc5(self.fc4(z)))
        def gumbel_loss(x, x_hat):
            q_y = tf.reshape(logits_y, (-1, N, M))
            q_y = tf.nn.softmax(q_y)
            log_q_y = tf.log(q_y + 1e-20)
            kl_tmp = q_y * (log_q_y - tf.log(1.0/M))
            KL = tf.reduce_sum(kl_tmp, axis=(1, 2))
            elbo = tf.reduce_sum(data_dim * tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_hat))
            elbo = elbo - KL 
            return tf.reduce_sum(elbo)

        argmax_y = tf.reduce_max(tf.reshape(logits_y, (-1, N, M)), axis=-1, keep_dims=True)
        self.argmax_y = tf.equal(tf.reshape(logits_y, (-1, N, M)), argmax_y)


        # sample from gaussian distribution
        '''
        eps = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.n_hidden]), 0, 1, dtype = tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.reconstruction = tf.add(tf.matmul(self.z, self.weights['w2']), self.weights['b2'])
        '''

        # cost
        self.cost = gumbel_loss(self.x, self.x_hat)
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
