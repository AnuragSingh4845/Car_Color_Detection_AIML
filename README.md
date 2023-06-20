# CAR COLOR DETECTION
This notebook contains a custom callback you may wish to copy and use
It is a combination of the Keras callbacks Reduce Learning Rate on Plateau,
Early Stopping and Model Checkpoint but eliminates some of the limitations
of each. In addition it provides an easier to read summary of the model's
performance at the end of each epoch. It also provides a handy feature
that enables you to set the number of epochs to train for until a message
asks if you wish to halt training on the current epoch by entering H or
to enter an integer which will determine how many more epochs to run
before the message appears again. This is very useful if you are training
a model and decide the metrics are satisfactory and you want to end
the model training early. Note the callback always returns your model
with the weights set to those of the epoch which had the highest performance
on the metric being monitored (accuracy or validation accuracy)
The callback initially monitors training accuracy and will adjust the learning
rate based on that until the accuracy reaches a user specified threshold
level. Once that level of training accuracy is achieved the callback switches
to monitoring validation loss and adjusts the learning rate based on that.
the callback is of the form:
callbacks=[LRA(model, base_model, patience, stop_patience, threshold,factor, dwell, batches, initial_epoch, epochs, ask_epoch )]
**where:**
  
* **model** is your compiled model
* **base**_model is the name of your base_model if you are doing transfer learning.
for example you might have in your model
base_model=tf.keras.applications.EfficientNetB1(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') base_model.trainabel=False During training you will be asked if you want to do fine tuning
If you enter F to the query, the base_model will be set to trainable by the callback If you are not doing transfer learning set base_model==None
* **patience** is an integer that determines many consecutive epochs can occur before the learning rate will be adjusted (similar to patience parameter in Reduce Learning Rate on Plateau)

* **stop_patience** is an integer that determines hom many consecutive epochs for which the learning rate was adjusted but no improvement in the monitored metric occurred before training is halted(similar to patience parameter in early stopping)

* **threshold** is a float that determines the level that training accuracy must achieve before the callback switches over to monitoring validation loss. This is useful for cases where the validation loss in early epochs tends to vary widely and can cause unwanted behavior when using the conventional Keras callbacks

* **factor** is a float that determines the new learning rate by the equation lr=lr*factor. (similar to the factor parameter in Reduce Learning Rate on Plateau)
* **dwell** is a boolean. It is used in the callback as part of an experiment on training models. If on a given epoch the metric being monitored fails to improve it means your model has moved to a location on the surface of Nspace (where N is the number of trainable parameters) that is NOT as favorable (poorer metric performance) than the position in Nspace you were in for the previous epoch. If dwell is set to True the callback loads the model with the weights from the previous (better metric value) epoch. Why move to a worse place if the place you were in previously was better. Then the learning rate is reduced for the next epoch of training. If dwell is set to false this action does not take place.
* **batches** is an integer. It should be set to a value of batches=int(number of traing samples/batch_size). During training the callback provides information during an epoch of the form 'processing batch of batches accuracy= accuracy loss= loss where batch is the current * **batch** being processs, batches is as described above, accuracy is the current training accuracy and loss is the current loss. Typically the message would appear as processing batch 25 of 50 accuracy: 54% loss: .04567. As each batch is processed these values are changed.
* **initial_epoch** is an integer. Typically set this to zero Itis used in the information printed out for each epoch. In the case where you train the model say with the basemodel weights frozen say you train for 10 epochs. Then you want to fine tune the model and train for more eppochs for the second training session you would reinstantiate the callback and set initial_epoch=10.
* **epochs** an integer value for the number of epochs to train
* **ask_epoch** is either set to an integer value or None. If set to an integer it denotes the epoch number at which user input is requested. If the user enter H training is halted. If the user inters an integer it represents how many more epochs to run before you are asked for the user input again. If the user enters F the base_model is made trainable If ask_epoch is set to None the user is NOT asked to provide any input. This feature is handy is when training your model and the metrics are either unsatisfactory and you want to stop training, or for the case where your metrics are satisfactory and there is no need to train any further. Note you model is always set to the weights for the epoch that had the beset metric performance. So if you halt the training you can still use the model for predictions.
** **Example of Use:**
callbacks=[LRA(model=my_model, base_model=base_model, patience=1,stop_patience=3,
threshold=.9, factor=.5, dwell=True,batches=85, initial_epoch=0 , epochs=20, ask_epoch=5)] this implies:

- your model is my_model
- base_model is the name of your base_model if you are doing transfer learning
- after 1 epoch of no improvement the learning rate will be reduced
- after 3 consecutive adjustment of the leaarning rate with no metric improve training terminates
- once the training accuracy reaches 90% the callback adjust learning rate based on validation loss
- when the learning rate is adjust the new learning rate is .5 X learning rate
- if the current epoch's metric value did not improve, the weights for the prior epoch are loaded and the learning rate is reduced
- 85 batches of data are run to complete an epoch
- the initial epoch is 0
- train for 20 epochs
- after the fifth epoch you will be asked if you want to halt training by entering H or enter an integer denoting how many more epochs to run before you will be prompted again or enter F to make the base_model=trainable


It uses the function trim to set the maximum number of samples in a class defined by the string column to max_samples. if the number of samples is less than min_samples the class is eliminated from the dataset. If some classes have less than max_samples, then augmented images are created for that class and stored in the working_dir so the class will have max_samples of images. After augmentation an aug_df is created for the augmented images in the working_dir. The aug_df is then merged with the original train_df to produce a new train_df that has exactly max_sample images in each class thus creating a balanced training set.

## MODULES

```sh
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
import pandas as pd
import shutil
import time
import cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import seaborn as sns
sns.set_style('darkgrid')
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from IPython.core.display import display, HTML
# stop annoying tensorflow warning messages
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
print ('modules loaded')
```
## DATASET

Images in this dataset were sourced from google. The were further cleaned up, post-processed, annotated and consolidated into a dataset. VCoR is a large scale and most diverse Vehicle color recognition dataset. VCoR contains 10k+ image samples and 15 color classes which is almost twice as diverse as the largest existing dataset. The 15 color categories represent the most popular vehicle color models according to CarMax, including: white, black, grey, silver, red, blue, brown, green, beige, orange, gold, yellow, purple, pink, and tan.

CONTENT
There is one main zip file available for download which contains 3 sub-folders.
1) train folder contains 15 folders for all 15 color classes, and about 7.5K images
2) val folder contains 15 folders for all 15 color classes, and about 1.5K images
3) test folder contains 15 folders for all 15 color classes, and about 1.5K images

LINKS TO DATA SETS ARE GIVE BELOW👇

[KAGGLE]([/guides/content/editing-an-existing-page](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset)https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset)

[GOOGLE DRIVE]([/guides/content/editing-an-existing-page](https://drive.google.com/file/d/1wpW7hD2ryxaeCuPYZCtUhzAs34QwM4o7/view?usp=sharing))
