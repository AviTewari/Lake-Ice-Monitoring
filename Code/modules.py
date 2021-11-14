import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from PIL import Image 

# set the necessary directories
img_dir = 'Data/Images'

img_filenames = os.listdir(img_dir)
img_names = [s.split('.')[0] for s in img_filenames]

img_ext = '.jpg'