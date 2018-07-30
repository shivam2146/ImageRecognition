import tensorflow as tf
import os, shutil
from ..im_conv_mnist import conv_mnist

def inference(x):
    W_init = tf.random_normal_initializer(stddev=0.5)
    W = tf.get_variable('W', [784,10], initializer = W_init)
    b_init = tf.constant_initializer(value = 0)
    b = tf.get_variable('b', [10], initializer = b_init)
    output = tf.nn.softmax(tf.matmul(x,W)+b)
    w_hist = tf.summary.histogram('weights',W)
    b_hist = tf.summary.histogram('biases', b)
    y_hist = tf.summary.histogram('output', output)
    return output


def check_digit_logistic(img_name):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, 784])
        output = inference(x)
        saver = tf.train.Saver()
        sess = tf.Session()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(os.path.dirname(base_dir), "logistic_regression/logistic_logs/")
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

        result = sess.run(tf.argmax(output,1), feed_dict={x: conv_mnist(img_name)})
        print("Result:", result)
        return result
