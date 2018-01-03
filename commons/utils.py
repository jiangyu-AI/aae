from scipy.misc import imsave
import numpy as np

def merge_gray_images(images):
    b, h, w = images.shape[0], images.shape[1], images.shape[2]
    size = np.sqrt(b)
    if size != int(size):
        raise ValueError('Unsupported input dimensions, need squared number of images')
    size = int(size)
    img = np.zeros((int(h * size), int(w * size)))
    for idx, image in enumerate(images):
        i = idx % size
        j = idx // size
        img[j*h:j*h+h, i*w:i*w+w] = image
    return img

def save_images_mnist(images, file_path):
    '''BHW'''
    img = merge_gray_images(images)
    imsave(file_path,img) 
    return file_path


def merge_rgb(images):
    b, h, w = images.shape[0], images.shape[1], images.shape[2]
    size = np.sqrt(b)
    if size != int(size):
        raise ValueError('Unsupported input dimensions, need squared number of images')
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def main():
    # test 
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/scratch/jy1367/data/mnist', one_hot=True)
    images,_ = mnist.train.next_batch(64)
    images=np.reshape(images,[-1,28,28])
    print(np.shape(images))
    save_images_mnist(images, '/scratch/jy1367/workspace/autoencoder/result/util_test.png')
