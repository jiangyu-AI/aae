'''PixelCNN Model'''

import tensorflow as tf
import numpy as np

def get_weights(shape, name, horizontal, mask_mode='noblind', mask=None):
    weights_initializer = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable(name, shape, tf.float32, weights_initializer)

    '''
        Use of masking to hide subsequent pixel values 
    '''
    if mask:
        filter_mid_y = shape[0]//2
        filter_mid_x = shape[1]//2
        mask_filter = np.ones(shape, dtype=np.float32)
        if mask_mode == 'noblind':
            if horizontal:
                # All rows after center must be zero
                mask_filter[filter_mid_y+1:, :, :, :] = 0.0
                # All columns after center in center row must be zero
                mask_filter[filter_mid_y, filter_mid_x+1:, :, :] = 0.0
            else:
                if mask == 'a':
                    # In the first layer, can ONLY access pixels above it
                    mask_filter[filter_mid_y:, :, :, :] = 0.0
                else:
                    # In the second layer, can access pixels above or even with it.
                    # Reason being that the pixels to the right or left of the current pixel
                    #  only have a receptive field of the layer above the current layer and up.
                    mask_filter[filter_mid_y+1:, :, :, :] = 0.0

            if mask == 'a':
                # Center must be zero in first layer
                mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.0
        else:
            mask_filter[filter_mid_y, filter_mid_x+1:, :, :] = 0.
            mask_filter[filter_mid_y+1:, :, :, :] = 0.

            if mask == 'a':
                mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.
                
        W *= mask_filter 
    return W

def get_bias(shape, name):
    return tf.get_variable(name, shape, tf.float32, tf.zeros_initializer)

def conv_op(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

class GatedCNN():
    def __init__(self, W_shape, fan_in, horizontal, gated=True, payload=None, mask=None, activation=True, conditional=None, conditional_image=None):
        self.fan_in = fan_in
        in_dim = self.fan_in.get_shape()[-1]#64
        self.W_shape = [W_shape[0], W_shape[1], in_dim, W_shape[2]]#W_shape=20  
        self.b_shape = W_shape[2]

        self.in_dim = in_dim
        self.payload = payload
        self.mask = mask
        self.activation = activation
        self.conditional = conditional
        self.conditional_image = conditional_image
        self.horizontal = horizontal
        
        if gated:
            self.gated_conv()
        else:
            self.simple_conv()

    def gated_conv(self):
        W_f = get_weights(self.W_shape, "v_W", self.horizontal, mask=self.mask)
        W_g = get_weights(self.W_shape, "h_W", self.horizontal, mask=self.mask)

        b_f_total = get_bias(self.b_shape, "v_b")
        b_g_total = get_bias(self.b_shape, "h_b")
        if self.conditional is not None:
            h_shape = int(self.conditional.get_shape()[1])
            V_f = get_weights([h_shape, self.W_shape[3]], "v_V", self.horizontal)
            b_f = tf.matmul(self.conditional, V_f)
            V_g = get_weights([h_shape, self.W_shape[3]], "h_V", self.horizontal)
            b_g = tf.matmul(self.conditional, V_g)

            b_f_shape = tf.shape(b_f)
            b_f = tf.reshape(b_f, (b_f_shape[0], 1, 1, b_f_shape[1]))
            b_g_shape = tf.shape(b_g)
            b_g = tf.reshape(b_g, (b_g_shape[0], 1, 1, b_g_shape[1]))

            b_f_total = b_f_total + b_f
            b_g_total = b_g_total + b_g
        if self.conditional_image is not None:
            b_f_total = b_f_total + tf.layers.conv2d(self.conditional_image, self.in_dim, 1, use_bias=False, name="ci_f")
            b_g_total = b_g_total + tf.layers.conv2d(self.conditional_image, self.in_dim, 1, use_bias=False, name="ci_g")

        conv_f = conv_op(self.fan_in, W_f)
        conv_g = conv_op(self.fan_in, W_g)
       
        if self.payload is not None:
            conv_f += self.payload
            conv_g += self.payload

        self.fan_out = tf.multiply(tf.tanh(conv_f + b_f_total), tf.sigmoid(conv_g + b_g_total))

    def simple_conv(self):
        W = get_weights(self.W_shape, "W", self.horizontal, mask_mode="standard", mask=self.mask)
        b = get_bias(self.b_shape, "b")
        conv = conv_op(self.fan_in, W)
        if self.activation: 
            self.fan_out = tf.nn.relu(tf.add(conv, b))
        else:
            self.fan_out = tf.add(conv, b)

    def output(self):
        return self.fan_out 


class PixelCNN(object):
    def __init__(self,
                 lr,
                 global_step,
                 grad_clip,
                 height, 
                 width,
                 channels,
                 #embeds, 
                 #K, 
                 num_classes, 
                 num_layers, 
                 num_maps,
                 is_training=True):
        self.X = tf.placeholder(tf.float32,[None,height,width,channels]) # _ * 28 * 28 * 1 for training MNIST, when training prior: _ * 4 * 4 * 3
        #self.X = tf.placeholder(tf.float32,[None,hidden_size,hidden_size,hidden_channel]) # when training prior: _ * 4 * 4 * 3
        onehot_h = None
        if( num_classes is not None ):
            self.h = tf.placeholder(tf.int32,[None,])
            onehot_h = tf.one_hot(self.h,num_classes,axis=-1)
        else:
            onehot_h = None
        '''
        if( embeds is not None ):
            X_processed = tf.gather(tf.stop_gradient(embeds),self.X) # _ * 3 * 3 * 64
        else:
            embeds = tf.get_variable('embed', [K,D],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
            X_processed = tf.gather(embeds,self.X)
        '''
        X_processed = self.X

        v_stack_in, h_stack_in = X_processed, X_processed
        for i in range(num_layers):
            filter_size = 3 if i > 0 else 7
            mask = 'b' if i > 0 else 'a'
            residual = True if i > 0 else False
            i = str(i)
            with tf.variable_scope("v_stack"+i):
                v_stack = GatedCNN([filter_size, filter_size, num_maps], v_stack_in, False, mask=mask, conditional=onehot_h).output()
                v_stack_in = v_stack
            with tf.variable_scope("v_stack_1"+i):
                v_stack_1 = GatedCNN([1, 1, num_maps], v_stack_in, False, gated=False, mask=mask).output()
            with tf.variable_scope("h_stack"+i):
                h_stack = GatedCNN([1, filter_size, num_maps], h_stack_in, True, payload=v_stack_1, mask=mask, conditional=onehot_h).output()

            with tf.variable_scope("h_stack_1"+i):
                h_stack_1 = GatedCNN([1, 1, num_maps], h_stack, True, gated=False, mask=mask).output()
                if residual:
                    h_stack_1 += h_stack_in # Residual connection
                h_stack_in = h_stack_1

        with tf.variable_scope("fc_1"):
            fc1 = GatedCNN([1, 1, 32], h_stack_in, True, gated=False, mask='b').output()
            #fc1 = GatedCNN([1, 1, conf.f_map], h_stack_in, True, gated=False, mask='b').output()

        #if conf.data == "mnist":
        with tf.variable_scope("fc_2"):
            self.fc2 = GatedCNN([1, 1, 3], fc1, True, gated=False, mask='b', activation=False).output()
            #self.fc2 = GatedCNN([1, 1, 3], fc1, True, gated=False, mask='b', activation=False).output()
        '''
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fc2, labels=self.X))
        '''
        self.pred = tf.nn.sigmoid(self.fc2)

        loss_per_batch_point = tf.reduce_sum(tf.square(self.fc2 - self.X),axis=3)
        self.loss = tf.reduce_mean(loss_per_batch_point, axis=[0,1,2])

        '''
        with tf.variable_scope("fc_1"):
            fc1 = GatedCNN([1, 1, num_maps], h_stack_in, True, gated=False, mask='b').output()

        with tf.variable_scope("fc_2"):
            self.fc2 = GatedCNN([1, 1, D], fc1, True, gated=False, mask='b', activation=False).output()
            #self.fc2 = GatedCNN([1, 1, K], fc1, True, gated=False, mask='b', activation=False).output()
            self.dist = tf.distributions.Categorical(logits=self.fc2)
            self.sampled = self.dist.sample()
            self.log_prob = self.dist.log_prob(self.sampled)

        loss_per_batch = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.fc2,
                                                                                      labels=self.X),axis=[1,2])
        self.loss = tf.reduce_mean(loss_per_batch,axis=0)
        '''

        save_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,tf.contrib.framework.get_name_scope())
        self.saver = tf.train.Saver(var_list=save_vars,max_to_keep = 3)

        if( is_training ):
            with tf.variable_scope('backward'):
                optimizer = tf.train.AdamOptimizer(lr)

                gradients = optimizer.compute_gradients(self.loss,var_list=save_vars)
                if( grad_clip is None ):
                    clipped_gradients = gradients
                else :
                    clipped_gradients = [(tf.clip_by_value(_[0], -grad_clip, grad_clip), _[1]) for _ in gradients]
                self.train_op = optimizer.apply_gradients(clipped_gradients,global_step)

    def _sample_from_prior(self,sess,classes,batch_size):
        size = self.X.get_shape()[1] # size of latents in x_axis or y_axis
        D = self.X.get_shape()[-1] # dim of latent
        feed_dict={
            self.X: np.zeros([len(classes)*batch_size,size,size,D],np.float32) # [10*10, 3,3] instead of  [10 * 10, 256, 256]
        }
        if( classes is not None ):
            feed_dict[self.h] = np.repeat(classes,batch_size).astype(np.int32) # 10 * 10

        #log_probs = np.zeros((len(classes)*batch_size,)) # 10 * 10
        for i in range(size):
            for j in range(size):
                for k in range(D):
                    sampled = sess.run([self.pred],feed_dict=feed_dict)
                    #sampled,log_prob = sess.run([self.pred,self.log_prob],feed_dict=feed_dict)
                    feed_dict[self.X][:,i,j,k]= sampled[:,i,j,k]
                    #log_probs += log_prob[:,i,j]
        return feed_dict[self.X]#, log_probs

    def sample_from_prior(self, sess, classes, batch_size):
        #print("Generating Sample Images...")
        size = self.X.get_shape()[1] # size of latents in x_axis or y_axis
        channels = self.X.get_shape()[-1] # dim of latent

        #n_row, n_col = 10,10
        samples = np.zeros((len(classes)*batch_size, size, size, channels), dtype=np.float32)
        data_dict = {self.X:samples}
        # TODO make it generic
        if classes is not None:
            data_dict[self.h] = np.repeat(classes,batch_size).astype(np.int32) # 10 * 10
        #labels = one_hot(np.array([0,1,2,3,4,5,6,7,8,9]*10), classes)
        for i in range(size):
            for j in range(size):
                for k in range(channels):
                    #if conf.conditional is True:
                    #    data_dict[h] = labels
                    next_sample = sess.run(self.pred, feed_dict=data_dict)
                    #if conf.data == "mnist":
                    #    next_sample = binarize(next_sample)
                    data_dict[self.X][:, i, j, k] = next_sample[:, i, j, k]
        return samples
        #save_images(samples, n_row, n_col, conf, suff)


    def generate_samples(sess, X, h, pred, conf, suff):
        print("Generating Sample Images...")
        n_row, n_col = 10,10
        samples = np.zeros((n_row*n_col, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
        # TODO make it generic
        labels = one_hot(np.array([0,1,2,3,4,5,6,7,8,9]*10), conf.num_classes)

        for i in range(conf.img_height):
            for j in range(conf.img_width):
                for k in range(conf.channel):
                    data_dict = {X:samples}
                    if conf.conditional is True:
                        data_dict[h] = labels
                    next_sample = sess.run(pred, feed_dict=data_dict)
                    if conf.data == "mnist":
                        next_sample = binarize(next_sample)
                    samples[:, i, j, k] = next_sample[:, i, j, k]
        save_images(samples, n_row, n_col, conf, suff)

    def save(self,sess,dir,step=None):
        if(step is not None):
            self.saver.save(sess,dir+'/model.ckpt',global_step=step)
        else :
            self.saver.save(sess,dir+'/last.ckpt')

    def load(self,sess,model):
        self.saver.restore(sess,model)

