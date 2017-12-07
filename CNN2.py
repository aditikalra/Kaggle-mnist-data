
# coding: utf-8

# In[22]:

import numpy as np
import pandas as pd

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

# settings
LEARNING_RATE = 1e-4
# set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 2500        
    
DROPOUT = 0.5
BATCH_SIZE = 50

# set to 0 to train on all available data
VALIDATION_SIZE = 2000

# image number to output
IMAGE_TO_DISPLAY = 10


# In[23]:

import os 
import pandas as pd
import numpy as np
import datetime 
import time
from multiprocessing import Pool
from functools import partial
import sys
import math
from sklearn.metrics import log_loss

import keras
import keras.backend as K
import theano.tensor as T
from keras.models import load_model

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import BatchNormalization,MaxPool2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, RMSprop
from  keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau


# In[24]:
path='./'
data = pd.read_csv(path+'train.csv')
print('data({0[0]},{0[1]})'.format(data.shape))
print (data.head())


# In[25]:

images = data.iloc[:,1:].values
images = images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)

print('images({0[0]},{0[1]})'.format(images.shape))


# In[26]:

image_size = images.shape[1]
print ('image_size => {0}'.format(image_size))

# in this case all images are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))


# In[27]:

# display image
def display(img):
    
    # (784) => (28,28)
    one_image = img.reshape(image_width,image_height)
    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)

# output image     
display(images[IMAGE_TO_DISPLAY])


# In[28]:

labels_flat = data[[0]].values.ravel()

print('labels_flat({0})'.format(len(labels_flat)))
print ('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels_flat[IMAGE_TO_DISPLAY]))

labels_count = np.unique(labels_flat).shape[0]

print('labels_count => {0}'.format(labels_count))


# In[29]:


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    print(num_labels)
    print(num_classes)
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    print(index_offset)
    print(labels_dense.ravel())
    print(index_offset + labels_dense.ravel())
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

print('labels({0[0]},{0[1]})'.format(labels.shape))
print ('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))


# In[30]:

print (labels.shape)
print (images.shape)


# In[31]:

images_reshape =images.reshape(42000,28,28,1)


model = Sequential()

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# In[32]:

datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)
model.compile(loss='categorical_crossentropy', optimizer =RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), metrics=["accuracy"])
#annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[ ]:

model.fit_generator(datagen.flow(images_reshape[6000:], labels[6000:], batch_size=50),
                           steps_per_epoch=images_reshape.shape[0] /50,
                           epochs=15, 
                           verbose=2,  
                           validation_data=(images_reshape[:6000,:], labels[:6000,:]),
                           callbacks=[learning_rate_reduction])


# In[33]:

test = pd.read_csv(path+'test.csv')


# convert from [0:255] => [0.0:1.0]
test_data = np.multiply(test.values, 1.0 / 255.0)

print('test_data({0[0]},{0[1]})'.format(test_data.shape))


# In[34]:

test_data =test_data.reshape(test_data.shape[0],28,28,1)
scores=model.predict(test_data)


# In[ ]:

sample_submission= pd.read_csv(path+"sample_submission.csv")


# In[ ]:

for index, row in sample_submission.iterrows():
    row['Label'] = np.argmax(scores[index])
    
print (sample_submission.Label.nunique())
sample_submission.to_csv(path+'final4.csv', index=False) 


# In[35]:

good_score = pd.read_csv(path+'final2.csv')


# In[36]:

X_train = images_reshape[4000:]
X_valid = images_reshape[:4000]
X_score = test_data
y_train = labels[4000:]
y_valid = labels[:4000]
y_score = good_score['Label']
y_score = np.array(pd.get_dummies(y_score))


# In[37]:

print(X_train.shape, y_train.shape)
print(X_score.shape, y_score.shape)
print(X_valid.shape, y_valid.shape)


# In[41]:

transform = {}
transform['width_shift_range'] = 0.075
transform['height_shift_range'] = 0.075
transform['rotation_range'] = 5
transform['shear_range'] = 0.3
transform['zoom_range'] = 0.075

generator = ImageDataGenerator(**transform)

train_batch = generator.flow(X_train, y_train, batch_size=20)
score_batch = generator.flow(X_score, y_score, batch_size=10)


model = Sequential()
model.add(BatchNormalization(input_shape=(28,28,1), axis=-1))
model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer =RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

def mix_iterator(iterators):
    while True:
        nexts = [next(iter_tuple) for iter_tuple in iterators]
        X = np.concatenate([n[0] for n in nexts])
        y = np.concatenate([n[1] for n in nexts])
        yield (X, y)
        
params = {}
params['epochs'] = 1
params['steps_per_epoch'] = 2200 # 36000 / 20
params['validation_data'] = (X_valid, y_valid)        
params['callbacks']=[learning_rate_reduction]


# In[ ]:

params['epochs'] = 15
model.fit_generator(mix_iterator([train_batch, score_batch]), verbose=2,**params)


# In[ ]:

test = pd.read_csv(path+'test.csv')


# convert from [0:255] => [0.0:1.0]
test_data = np.multiply(test.values, 1.0 / 255.0)
test_data =test_data.reshape(test_data.shape[0],28,28,1)

print('test_data({0[0]},{0[1]})'.format(test_data.shape))
scores=model.predict(test_data)

sample_submission= pd.read_csv(path+"sample_submission.csv")

for index, row in sample_submission.iterrows():
    row['Label'] = np.argmax(scores[index])
    
print (sample_submission.Label.nunique())
sample_submission.to_csv(path+'final4_new.csv', index=False) 

