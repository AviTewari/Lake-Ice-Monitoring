from modules import *
from masks import *
from image_preprocessing import *

def display(display_list):
    fig, axs = plt.subplots(nrows=1, ncols = len(display_list), figsize=(15, 6))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        axs[i].set_title(title[i])
        if i==0:
            axs[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            msk = axs[i].imshow(display_list[i], cmap = cmap, vmin=0, vmax=n_colors-1)
        axs[i].axis('off')
        
    #plot colorbar
    cbar = fig.colorbar(msk, ax=axs, location='right')
    tick_locs = (np.arange(n_colors) + 0.5)*(n_colors-1)/n_colors#new tick locations so they are in the middle of the colorbar
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(np.arange(n_colors))
    plt.show()

for image, mask in ds_train.take(1):
    sample_image, sample_mask = image, mask
display([sample_image, sample_mask])