import tensorflow as tf
import time, os, sys
from ..im_conv_mnist import conv_mnist
# Architecture
n_hidden_1 = 256
n_hidden_2 = 256

def conv2d(input, weight_shape, bias_shape):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b))

def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)


def inference(x, keep_prob):

    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5, 5, 1, 32], [32])
        pool_1 = max_pool(conv_1)

    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [5, 5, 32, 64], [64])
        pool_2 = max_pool(conv_2)

    with tf.variable_scope("fc"):
        pool_2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64])
        fc_1 = layer(pool_2_flat, [7*7*64, 1024], [1024])

        # apply dropout
        fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    with tf.variable_scope("output"):
        output = layer(fc_1_drop, [1024, 10], [10])

    return output

def check_digit_cnn(img_name):

    with tf.device("/device:gpu:0"):

        with tf.Graph().as_default():

            with tf.variable_scope("mnist_conv_model"):

                x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
                keep_prob = tf.placeholder(tf.float32) # dropout probability
                output = inference(x, keep_prob)
                saver = tf.train.Saver()
                sess = tf.Session()
                base_dir = os.path.dirname(os.path.abspath(__file__))
                checkpoint_path = os.path.join(os.path.dirname(base_dir), "cnn/conv_mnist_logs/")
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

                result = sess.run(tf.argmax(output,1), feed_dict={x: conv_mnist(img_name), keep_prob: 1})
                print("Result:", result)
                return result
