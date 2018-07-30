import tensorflow as tf
import os, shutil
from ..im_conv_mnist import conv_mnist

n_hidden_1 = 256
n_hidden_2 = 256

def layer(x, w_shape, b_shape):
    W_init = tf.random_normal_initializer(stddev=(2.0/w_shape[0])**0.5)
    W = tf.get_variable('W', w_shape, initializer = W_init)
    b_init = tf.constant_initializer(value = 0)
    b = tf.get_variable('b', b_shape, initializer = b_init)
    output = tf.nn.relu(tf.matmul(x,W)+b)
    w_hist = tf.summary.histogram('weights',W)
    b_hist = tf.summary.histogram('biases', b)
    y_hist = tf.summary.histogram('output', output)
    return output

def inference(x):
    with tf.variable_scope('layer_1'):
        output_1 = layer(x, [784,n_hidden_1],[n_hidden_1])
    with tf.variable_scope('layer_2'):
        output_2 = layer(output_1, [n_hidden_1,n_hidden_2], [n_hidden_2])
    with tf.variable_scope('output_layer'):
        output = layer(output_2, [n_hidden_2,10], [10])
    return tf.nn.softmax(output)

def check_digit_mlp(img_name):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, 784])
        output = inference(x)
        saver = tf.train.Saver()
        sess = tf.Session()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(os.path.dirname(base_dir), "multilayer/mlp_logs/")
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        result = sess.run(tf.argmax(output,1), feed_dict={x: conv_mnist(img_name)})
        print("Result:", result)
        return result
