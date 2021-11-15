from modules import *
from UNet import *
from checkpoints import *

model=get_unet()
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics=['sparse_categorical_accuracy', UpdatedMeanIoU(num_classes=n_colors)])

EPOCHS = 100
VAL_SUBSPLITS = 5
VALIDATION_STEPS = VAL_LENGTH//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, 
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=val_dataset,
                          callbacks=[DisplayCallback(), lr_callback, cp_callback])

'''
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

ax[0].plot(epochs, loss, 'r', label='Training')
ax[0].plot(epochs, val_loss, 'bo', label='Validation')
ax[0].set_title('Training and Validation Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss Value')
ax[0].legend()

IoU_key = list(model_history.history.keys())[2]
acc = model_history.history[IoU_key]
val_acc = model_history.history['val_'+IoU_key]

ax[1].plot(epochs, acc, 'r', label='Training')
ax[1].plot(epochs, val_acc, 'bo', label='Validation')
ax[1].set_title('Training and Validation IoU')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('IoU Value')
ax[1].legend()
plt.show()

'''