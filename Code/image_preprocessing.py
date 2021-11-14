from modules import *
from mask_preprocessing import *
from oversampling import *
from read_images import *
from image_augmentation import *

class_max = mask_df.iloc[:,6:-1].idxmax(axis=1) #category of the most common class in the image. We will stratify our train test split by this
class_max.value_counts()


names = mask_df['name'].values
train_names, validation_names, train_max, validation_max = train_test_split(img_names, class_max, 
                                                                            train_size=0.8, test_size=0.2, 
                                                                            random_state=0, stratify=class_max)

#add over-sampled images to the train dataset
train_over_sample_names = np.array([name for name in train_names if name in over_sample_names])
N_over_sample = int(len(train_names)/1.5) #number of additional samples to add
ids = np.arange(len(train_over_sample_names))
choices = np.random.choice(ids, N_over_sample)#an additional set of images to add on to the train names
add_train_names = train_over_sample_names[choices].tolist()
train_names.extend(add_train_names)

IMG_SIZE = (256, 256)
TRAIN_LENGTH = int(len(train_names))
VAL_LENGTH = int(len(validation_names))
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

ds_train = tf.data.Dataset.from_tensor_slices((train_names))#read filenames
ds_train = ds_train.map(read_image, num_parallel_calls=tf.data.AUTOTUNE) #convert filenames to stream of images/masks
ds_train = ds_train.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE) #convert filenames to stream of images/masks
train_dataset = ds_train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

ds_val = tf.data.Dataset.from_tensor_slices((validation_names))#read filenames
ds_val = ds_val.map(read_image) #convert filenames to stream of images/masks
val_dataset = ds_val.batch(BATCH_SIZE)