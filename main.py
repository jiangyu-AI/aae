from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pathlib
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
from tqdm import tqdm
from six.moves import xrange

from autoencoder_models.ConvAutoencoder import ConvAutoencoder
from autoregressive_models.pixelcnn import PixelCNN
from commons.utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
  '--epochs',
  type=int,
  default=1,
  #default=30,
  help='Number of epochs to train.')

parser.add_argument(
    '--train_num',
    type=int,
    default=6,#0000,
    help='200000 for cifar10, 60000 for mnist, num of trianing examples in one epoch')

parser.add_argument(
  '--lr',
  type=float,
  default=0.001,
  #default=0.0001,
  help='learning rate')

parser.add_argument(
  '--num_samples',
  type=int,
  default=64,
  #default=64,
  help='Number of images in each result.')

parser.add_argument(
  '--batch_size',
  type=int,
  default=128,
  #default=64,
  help='Number of images in each batch.')

parser.add_argument(
  '--model_dir',
  type=str,
  default='./model_autoencoder',
  #default='/scratch/jy1367/workspace/autoencoder/model',
  help='directory where the trainded model is to be saved.')

parser.add_argument(
  '--model_pixelcnn_dir',
  type=str,
  default='./model_pixelcnn',
  #default='/scratch/jy1367/workspace/autoencoder/model',
  help='directory where the trainded model is to be saved.')

parser.add_argument(
  '--results_dir',
  type=str,
  default='./result',
  #default='/home/jy1367/workspace/autoencoder/result',
  help='directory where the trainded model is to be saved.')



parser.add_argument(
    '--data_set',
    type=str,
    default='mnist',
    help='Dataset to use, the other one is cifar10')

parser.add_argument(
    '--decay_steps',
    type=int,
    default=20000,
    help='Number of images to process in a batch')

parser.add_argument(
    '--decay_val',
    type=float,
    default=1.0,
    help='Number of images to process in a batch')

parser.add_argument(
    '--decay_staircase',
    type=bool,
    default=False,
    help='Number of images to process in a batch')

parser.add_argument(
    '--log_dir',
    type=str,
    default='./log',
    help='Data set i.e. input data to train on')

parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.0002,
    help='Data set i.e. input data to train on')

parser.add_argument(
    '--hidden_channel',
    type=int,
    default=3,
    #default=64,
    help='64 for mnist, 128 for cifar10. Embedding vector dimension')

parser.add_argument(
    '--grad_clip',
    type=float,
    default=1.0,
    help='Please set 1.0 for mnist, 5.0 for cifar10')

parser.add_argument(
    '--num_layers',
    type=int,
    default=12,
    help='Time interval to summary')

parser.add_argument(
    '--num_feature_maps',
    type=int,
    default=32,
    help='32 for mnist, 64 for cifar10')

parser.add_argument(
    '--summary_interval',
    type=int,
    default=1000,
    help='Time interval to summary')

parser.add_argument(
    '--save_interval',
    type=int,
    default=10000,
    help='Time interval to save model')

parser.add_argument(
    '--random_seed',
    type=int,
    default=1,
    help='Graph level random seed')


parser.add_argument(
    '--datasets_dir',
    type=str,
    default='/scratch/jy1367/datasets/mnist',
    help='Graph level random seed')


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


def train_autoencoder():
    n_samples = int(mnist.train.num_examples)

    training_epochs = FLAGS.epochs 
    #training_epochs = 20
    batch_size = FLAGS.batch_size
    display_step = 1

    autoencoder = ConvAutoencoder(
        n_input=784,
        n_hidden=200,
        transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.lr))

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)

            # Fit training using batch data 
            cost = autoencoder.partial_fit(batch_xs) 
            # Compute average loss 
            avg_cost += cost / n_samples * batch_size 
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%d,' % (epoch + 1),
                  "Cost:", "{:.9f}".format(avg_cost))

    print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))

    pathlib.Path(FLAGS.model_dir).mkdir(parents=True, exist_ok=True) 
    save_path = autoencoder.save(FLAGS.model_dir)
    print("Model saved in file: %s" % save_path)

def get_samples_autoencoder():
    # restore model 
    tf.reset_default_graph()
    autoencoder = ConvAutoencoder(
        n_input=784,
        n_hidden=200,
        transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.lr),
        is_training=False)
    autoencoder.restore(FLAGS.model_dir+'/model.ckpt')
    #get results
    not_used = get_random_block_from_data(X_train, FLAGS.num_samples)
    hidden = np.random.normal(size=[FLAGS.num_samples, 4,4,3])
    samples = autoencoder.generate(not_used, hidden)
    samples = np.reshape(samples,[-1,28,28])
    pathlib.Path(FLAGS.results_dir).mkdir(parents=True, exist_ok=True) 
    save_images_mnist(samples, FLAGS.results_dir+'/samples_autoencoder.png')
    print("Result saved in file: %s" % FLAGS.results_dir)

def transform_images2hidden():
    # restore model 
    tf.reset_default_graph()
    autoencoder = ConvAutoencoder(
        n_input=784,
        n_hidden=200,
        transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.lr))
    autoencoder.restore(FLAGS.model_dir+'/model.ckpt')
    xs,ys = mnist.train.images, mnist.train.labels
    hiddens = []
    for i in range(0,len(xs),FLAGS.batch_size):
        batch = xs[i:i+FLAGS.batch_size]
        hidden = autoencoder.transform(batch)
        hiddens.append(hidden)
    hiddens = np.concatenate(hiddens,axis=0)
    np.savez(os.path.join(os.path.dirname(FLAGS.model_dir),'hiddens_ys.npz'),hiddens=hiddens,ys=ys) # [3*3] indices of latent represetations embeddings and corresponding labels
    #test
    not_used = get_random_block_from_data(X_train, FLAGS.num_samples)
    orig = get_random_block_from_data(X_test,FLAGS.num_samples)
    recons = autoencoder.reconstruct(orig)
    orig = np.reshape(orig,[-1,28,28])
    recons = np.reshape(recons,[-1,28,28])
    pathlib.Path(FLAGS.results_dir).mkdir(parents=True, exist_ok=True) 
    save_images_mnist(orig, FLAGS.results_dir+'/origin.png')
    save_images_mnist(recons, FLAGS.results_dir+'/reconstruction.png')

class Latent_data():
    def __init__(self,path,validation_size=1):
    #def __init__(self,path,validation_size=5000):
        from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
        from tensorflow.contrib.learn.python.learn.datasets import base

        data = np.load(path)
        train = DataSet(data['hiddens'][validation_size:], data['ys'][validation_size:],reshape=False,dtype=np.uint8,one_hot=False) 
        validation = DataSet(data['hiddens'][:validation_size], data['ys'][:validation_size],reshape=False,dtype=np.uint8,one_hot=False)
        self.size = data['hiddens'].shape[1]
        self.data = base.Datasets(train=train, validation=validation, test=None)

def decode_latent(latents, save_file_path):
    # restore model 
    tf.reset_default_graph()
    autoencoder = ConvAutoencoder(
        n_input=784,
        n_hidden=200,
        transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.lr),
        is_training=False)
    autoencoder.restore(FLAGS.model_dir+'/model.ckpt')

   #get results
    not_used = get_random_block_from_data(X_train, len(latents))
    #not_used = get_random_block_from_data(X_train, FLAGS.num_samples)
    #hidden = np.random.normal(size=[FLAGS.num_samples, 4,4,3])
    #samples = autoencoder.generate(not_used, hidden)
    #samples = autoencoder.generate(not_used, batch_xs)
    samples = autoencoder.generate(not_used, latents)
    samples = np.reshape(samples,[-1,28,28])
    save_images_mnist(samples, save_file_path)
    print("Result saved in file: %s" % FLAGS.results_dir)

def decode_latent_test():
    latent_data = Latent_data(os.path.join(os.path.dirname(FLAGS.model_dir),'hiddens_ys.npz'))
    #batch_xs, batch_ys = latent_data.data.train.next_batch(FLAGS.num_samples)
    latents, testbatch_ys = latent_data.data.train.next_batch(64)
    save_images_mnist(latents[:,:,0], FLAGS.results_dir+'/latents_0.png')
    save_images_mnist(latents[:,:,1], FLAGS.results_dir+'/latents_1.png')
    save_images_mnist(latents[:,:,2], FLAGS.results_dir+'/latents_2.png')
    decode_latent(latents, FLAGS.results_dir+'/decode_latent_test.png')


def train_pixelcnn( data_set,
                    random_seed,
                    model_dir,
                    train_num,
                    batch_size,
                    learning_rate,
                    decay_val,
                    decay_steps,
                    decay_staircase,
                    grad_clip,
                    #K,
                    hidden_channel,#D,
                    #beta,
                    num_layers,
                    num_feature_maps,
                    summary_interval,
                    save_interval):
    
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    #log_dir = os.path.join(os.path.dirname(model_dir),'model_pixelcnn')
    if data_set == 'mnist':
        input_height = 24
        input_width = 24
        input_channel = 1
    elif data_set == 'cifar10':
        input_height = 32
        input_width = 32
        input_channel =3

    latent_data = Latent_data(os.path.join(os.path.dirname(model_dir),'hiddens_ys.npz'))

    # model_dir for Generate Images
    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        _not_used = tf.placeholder(tf.float32,[None,input_height,input_width,input_channel])# 32*32*3,24*24*1
        #vq_net = VQVAE(None,None,beta,_not_used,K,D,data_set,params,False)

    # model_dir for Training Prior
    with tf.variable_scope('pixelcnn'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = learning_rate
        height=4
        width=4
        channels=3
        num_classes=10
        net = PixelCNN(learning_rate,global_step,grad_clip,height,width,channels,num_classes,num_layers,num_feature_maps)
        #net = PixelCNN(learning_rate,global_step,grad_clip,latent_data.size,hidden_channel,10,num_layers,num_feature_maps)
    with tf.variable_scope('misc'):
        tf.summary.scalar('loss',net.loss)
        summary_op = tf.summary.merge_all()
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sample_images = tf.placeholder(tf.float32,[None,input_height,input_width,input_channel])#24*24*1,32*32*3
        sample_summary_op = tf.summary.image('samples',sample_images,max_outputs=20)

    # train
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init_op)
    #q_net.load(sess,model_dir)
    summary_writer = tf.summary.FileWriter(FLAGS.model_pixelcnn_dir,sess.graph)
    for step in tqdm(xrange(train_num),dynamic_ncols=True):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #batch_xs = np.reshape(batch_xs,[-1,28,28,1])
        batch_xs, batch_ys = latent_data.data.train.next_batch(batch_size)
        it,loss,_ = sess.run([global_step,net.loss,net.train_op],feed_dict={net.X:batch_xs,net.h:batch_ys})
        if( it % save_interval == 0 ):
            net.save(sess,FLAGS.model_pixelcnn_dir,step=it)
        if( it % summary_interval == 0 ):
            tqdm.write('[%5d] Loss: %1.3f'%(it,loss))
            summary = sess.run(summary_op,feed_dict={net.X:batch_xs,net.h:batch_ys})
            summary_writer.add_summary(summary,it)
        '''
        if( it % (summary_interval * 2) == 0 ):
            sampled_zs,log_probs = net.sample_from_prior(sess,np.arange(10),2)
            sampled_ims = sess.run(vq_net.gen,feed_dict={vq_net.latent:sampled_zs})
            summary_writer.add_summary(
                sess.run(sample_summary_op,feed_dict={sample_images:sampled_ims}),it)
        '''
    net.save(sess,FLAGS.model_pixelcnn_dir)

def restore_pixelcnn_and_sample(
                                learning_rate,
                                grad_clip,
                                num_layers,
                                num_feature_maps,
                                ):
    tf.reset_default_graph()
    #pixelcnn = PixelCNN(learning_rate,global_step,grad_clip,latent_data.size,hidden_channel,10,num_layers,num_feature_maps)
    global_step = tf.Variable(0, trainable=False)
    with tf.variable_scope('pixelcnn'):
        pixelcnn = PixelCNN(lr=FLAGS.learning_rate,global_step=global_step,grad_clip=FLAGS.grad_clip,height=4,width=4,channels=3,num_classes=10,num_layers=FLAGS.num_layers,num_maps=FLAGS.num_feature_maps,is_training=False)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    #sess.graph.finalize()
    sess.run(init_op)
    pixelcnn.load(sess,FLAGS.model_pixelcnn_dir+'/last.ckpt')
    sys.exit()
    # generate samples
    samples_latent = pixelcnn.sample_from_prior(sess,np.arange(10),10)
    #print(len(sampled_zs))
    #print(len(sampled_zs[0]))
    #print(sampled_zs)
    #samples_ = np.reshape(samples,[-1,28,28])
    save_images_mnist(samples_latent[:,:,0], FLAGS.results_dir+'/samples_latent_0.png')
    save_images_mnist(samples_latent[:,:,1], FLAGS.results_dir+'/samples_latent_1.png')
    save_images_mnist(samples_latent[:,:,2], FLAGS.results_dir+'/samples_latent_2.png')
    print("Result saved in file: %s" % FLAGS.results_dir)
    decode_latent(samples_latent, FLAGS.results_dir+'/samples.png')




if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    mnist = input_data.read_data_sets(FLAGS.datasets_dir, one_hot=False)
    X_train, X_test = mnist.train.images, mnist.test.images
    #X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

    train_autoencoder()
    #get_samples_autoencoder()
    transform_images2hidden()
    #decode_latent_test()
    train_pixelcnn( FLAGS.data_set,
                    FLAGS.random_seed,
                    FLAGS.model_dir,
                    FLAGS.train_num,
                    FLAGS.batch_size,
                    FLAGS.learning_rate,
                    FLAGS.decay_val,
                    FLAGS.decay_steps,
                    FLAGS.decay_staircase,
                    FLAGS.grad_clip,
                    FLAGS.hidden_channel,
                    FLAGS.num_layers,
                    FLAGS.num_feature_maps,
                    FLAGS.summary_interval,
                    FLAGS.save_interval)
    
    restore_pixelcnn_and_sample(FLAGS.learning_rate,
                                FLAGS.grad_clip,
                                FLAGS.num_layers,
                                FLAGS.num_feature_maps)
                                
