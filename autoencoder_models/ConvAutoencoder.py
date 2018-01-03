import tensorflow as tf
import sys
import numpy as np
import math

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

class ConvAutoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), is_training=True):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        #network_weights = self._initialize_weights()
        #self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.x_orig = self.x
        
        n_filters = [1,3,3,3]
        #n_filters = [1,10,10,10]
        # reshape to 2-d square tensor
        x = self.x
        if len(x.get_shape()) == 2:
            x_dim = np.sqrt(x.get_shape().as_list()[1])
            if x_dim != int(x_dim):
                raise ValueError('Unsupported input dimensions, need squared size')
            x_dim = int(x_dim)
            x_tensor = tf.reshape(x, [-1, x_dim, x_dim, n_filters[0]])
        elif len(x.get_shape()) == 4:
            x_tensor = x
        else:
            raise ValueError('Unsupported input dimensions')
        current_input = x_tensor
        orig_input = x_tensor

        def lrelu(x, leak=0.2, name='lrelu'):
            with tf.variable_scope(name):
                f1 = 0.5 * (1 + leak)
                f2 = 0.5 * (1 - leak)
                return f1 * x + f2 * abs(x)

        D = 4
        enc_spec = [
            Conv2d('conv2d_1',1,D//4,data_format='NHWC'), #d//4=16
            lambda t,**kwargs : tf.nn.relu(t),
            Conv2d('conv2d_2',D//4,D//2,data_format='NHWC'), # 
            lambda t,**kwargs : tf.nn.relu(t),
            Conv2d('conv2d_3',D//2,D,data_format='NHWC'), #d=64
            lambda t,**kwargs : tf.nn.relu(t),
        ]
        dc_spec = [
            TransposedConv2d('tconv2d_1',D,D//2,data_format='NHWC'),
            lambda t,**kwargs : tf.nn.relu(t),
            TransposedConv2d('tconv2d_2',D//2,D//4,data_format='NHWC'),
            lambda t,**kwargs : tf.nn.relu(t),
            TransposedConv2d('tconv2d_3',D//4,1,data_format='NHWC'),
            lambda t,**kwargs : tf.nn.sigmoid(t),
        ]
            
        # encoder
        filter_sizes = [3,3,3,3]
        encoder = []
        shapes = []
        for layer_i, n_output in enumerate(n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            shapes.append(current_input.get_shape().as_list())
            W = tf.Variable(
                tf.random_uniform([
                    filter_sizes[layer_i],
                    filter_sizes[layer_i],
                    n_input, n_output],
                    -1.0 / math.sqrt(n_input),
                    1.0 / math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([n_output]))
            encoder.append(W)
            #output = lrelu(
            output = tf.nn.relu(
                tf.add(tf.nn.conv2d(
                    current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        # store the latent representation
        if not is_training:
            self.hidden_input = tf.placeholder(tf.float32, [None, 4,4,3])
            current_input = self.hidden_input
        #self.hidden = current_input# _ * 4 * 4 * 3
            #self.hidden = hidden_input

        self.hidden = current_input# _ * 4 * 4 * 3
        #z = current_input
        #current_input = self.hidden
        encoder.reverse()
        shapes.reverse()

        # Build the decoder using the same weights
        for layer_i, shape in enumerate(shapes):
            W = encoder[layer_i]
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
            output = lrelu(tf.add(
                tf.nn.conv2d_transpose(
                    current_input, W,
                    tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                    strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        # %%
        # now have the reconstruction through the network
        y = current_input
        #self.reconstruction = self.x
        #self.reconstruction = tf.reshape(self.x, [-1,self.n_input])
        #self.reconstruction = tf.reshape(orig_input, [-1,self.n_input])
        self.reconstruction = tf.reshape(y, [-1,self.n_input])

        # cost function measures pixel-wise difference
        self.cost = tf.reduce_sum(tf.square(y - x_tensor))

        '''
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        '''

        # cost
        #self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        #tf.reset_default_graph()  # uncomment it to run in the jupyter notebook
        self.sess = tf.Session()
        self.sess.run(init)

        self.saver = tf.train.Saver()

    '''
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights
    '''

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        feed_dict = {self.x: X}
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, X, hidden = None):
        feed_dict = {self.x: X}
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([4, 4,4,3]))
        feed_dict[self.hidden_input] = hidden
        return self.sess.run(self.reconstruction, feed_dict=feed_dict)
        #return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        feed_dict = {self.x: X}
        #feed_dict[self.hidden_input] = hidden
        #print(self.sess.run(self.hidden, feed_dict=feed_dict))
        #sys.exit()
        return self.sess.run(self.reconstruction, feed_dict=feed_dict)

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

    def save(self, model_dir):
        return self.saver.save(self.sess, model_dir + '/model.ckpt')

    def restore(self, saved_model):
        return self.saver.restore(self.sess, saved_model)

    def return_x(self, X):
        return self.sess.run(self.x, feed_dict = {self.x: X})

    def return_x_orig(self, X):
        return self.sess.run(self.x_orig, feed_dict = {self.x: X})
