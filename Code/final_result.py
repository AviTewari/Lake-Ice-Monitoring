from modules import *
from UNet import *
from plot import *
from checkpoints import *

model.load_weights(checkpoint_dir + '/cp-0051.ckpt')
scores = model.evaluate(val_dataset, verbose=0)
print('Final Model Validation Scores')
print('Loss: {:.3f}'.format(scores[0]))
print('Accuracy: {:.3f}'.format(scores[1]))
print('IoU: {:.3f}'.format(scores[2]))

show_predictions(val_dataset, num=10, ele=3)

from sklearn.metrics import confusion_matrix
import seaborn as sns

def get_cm(model, val_ds):
    cm = np.zeros((8,8))
    for img_batch, mask_batch in val_dataset:
        y_pred = []
        y_true = []
        pred_batch = model.predict(img_batch)
        pred_batch = tf.argmax(pred_batch, axis=-1)#take the highest probability as the prediction for each pixel
        for n, pred in enumerate(pred_batch):
            pred = np.array(pred).flatten() #flattened array of predicted pixels for each image
            mask = np.array(mask_batch[n, ...]).flatten() #flattened array of mask pixels for the image
            y_pred.extend(pred)
            y_true.extend(mask)
        cm = cm + confusion_matrix(y_true, y_pred)
    return cm

cm = get_cm(model, val_dataset)
plt.figure(figsize=(12,8))
sns.heatmap(cm.astype(int), annot=True, fmt="d")
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()