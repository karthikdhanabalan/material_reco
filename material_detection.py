
# coding: utf-8

# In[76]:


import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D

import pickle


# In[77]:


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)


# In[78]:


X = tf.keras.utils.normalize(X)
y = tf.keras.utils.normalize(y)
keras.utils.to_categorical(X,num_classes = 10)
#keras.utils.to_categorical(y,num_classes = 10)


# In[79]:


X.shape


# In[113]:


model = Sequential()

#model.add(Conv2D(32, 3,data_format="channels_last", input_shape=(80,80,3)))
#model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation='relu', input_shape=(224,224,3)))
model.add(Conv2D(32,(2,2), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(GlobalAveragePooling2D())
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#model.add(Dropout(0.5))

model.add(Dense(15))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


# In[115]:


model.fit(X, y, batch_size=10, epochs=3, validation_split=0.3)

