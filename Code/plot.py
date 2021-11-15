from modules import *
from UNet import *
from show_image_and_mask import *


def create_mask(pred_mask, ele=0):
    pred_mask = tf.argmax(pred_mask, axis=-1)#use the highest proabbaility class as the prediction
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[ele]

#helper functions to plot image, mask, and predicted mask while training
def show_predictions(dataset=None, num=1, ele=0):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[ele], mask[ele], create_mask(pred_mask, ele)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])

#function to display loss during training
def plot_loss_acc(loss, val_loss, epoch):#, acc, val_acc, epoch):
    
    epochs = range(epoch+1)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

    ax.plot(epochs, loss, 'r', label='Training loss')
    ax.plot(epochs, val_loss, 'bo', label='Validation loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.legend()
    plt.show()