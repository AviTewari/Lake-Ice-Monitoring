from modules import *
from image_preprocessing import *
from mask_preprocessing import *

IMG_SIZE = (256, 256)
def read_image(image_name):
    image = tf.io.read_file(img_dir + image_name + img_ext)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    
    mask = tf.io.read_file(mask_dir + image_name + mask_ext)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, IMG_SIZE)
    mask = tf.cast(mask, tf.uint8)
    return image, mask