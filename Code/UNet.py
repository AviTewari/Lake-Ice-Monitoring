from modules import *
from image_preprocessing import *
from masks import *

def get_unet():
    inputs = Input(shape=[IMG_SIZE[0], IMG_SIZE[1], 3])
    conv1 = Conv2D(32, 3, 1, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, 1, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.5)(pool1)

    conv2 = Conv2D(64, 3, 1, activation='relu', padding='same')(drop1)
    conv2 = Conv2D(64, 3, 1, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(128, 3, 1, activation='relu', padding='same')(drop2)
    conv3 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(256, 3, 1, activation='relu', padding='same')(drop3)
    conv4 = Conv2D(256, 3, 1, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(512, 3, 1, activation='relu', padding='same')(drop4)
    conv5 = Conv2D(512, 3, 1, activation='relu', padding='same')(conv5)

    up6 = Conv2D(256, 3, activation = 'relu', padding = 'same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([up6, conv4], axis=3)
    drop6 = Dropout(0.5)(merge6)
    conv6 = Conv2D(256, 3, 1, activation='relu', padding='same')(drop6)
    conv6 = Conv2D(256, 3, 1, activation='relu', padding='same')(conv6)
    
    up7 = Conv2D(128, 3, activation = 'relu', padding = 'same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([up7, conv3], axis=3)
    drop7 = Dropout(0.5)(merge7)
    conv7 = Conv2D(128, 3, 1, activation='relu', padding='same')(drop7)
    conv7 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv7)
    
    up8 = Conv2D(64, 3, activation = 'relu', padding = 'same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([up8, conv2], axis=3)
    drop8 = Dropout(0.5)(merge8)
    conv8 = Conv2D(64, 3, 1, activation='relu', padding='same')(drop8)
    conv8 = Conv2D(64, 3, 1, activation='relu', padding='same')(conv8)
    
    up9 = Conv2D(32, 3, activation = 'relu', padding = 'same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([up9, conv1], axis=3)
    drop9 = Dropout(0.5)(merge9)
    conv9 = Conv2D(32, 3, 1, activation='relu', padding='same')(drop9)
    conv9 = Conv2D(32, 3, 1, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n_colors, 1, 1, activation='softmax')(conv9) #softmax converts the output to a list of probabilities that must sum to 1

    model = Model(inputs=inputs, outputs=conv10)
    return model

model = get_unet() 

# tf.keras.utils.plot_model(model, show_shapes=True)