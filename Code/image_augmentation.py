from modules import *
from image_preprocessing import *
from read_images import * 
from mask_preprocessing import *

def augment_image(image, mask):
    n = tf.random.uniform([], 0,1)
    if n<0.5: 
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
        
    n = tf.random.uniform([], 0,1)
    if n<0.5: 
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    
    #rotate image randomly in the range of +-5 degrees
    n = tf.random.uniform([], -1,1)
    image = tfa.image.rotate(image, np.pi/36*n, fill_mode='constant', fill_value=0)#add black to rotated corners
    mask = tfa.image.rotate(mask, np.pi/36*n, fill_mode='constant', fill_value=7)#make this black space correspond to land
    return image, mask