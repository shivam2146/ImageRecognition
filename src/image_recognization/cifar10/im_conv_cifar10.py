import tensorflow as tf
from PIL import Image

NUM_CLASSES = 10
IMAGE_SIZE = 24

def im_cov_cifar10(image):
    reshaped_image = tf.cast(image, tf.float32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)
    float_image = tf.image.per_image_standardization(resized_image)
    image = Image(float_image)
    image.show()
