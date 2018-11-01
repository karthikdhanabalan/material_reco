
# coding: utf-8

# In[59]:


import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
import numpy as np
import pickle
from tensorflow.keras.callbacks import TensorBoard


# In[76]:


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)


# In[77]:


#X = tf.keras.utils.normalize(X)
#y = tf.keras.utils.normalize(y)
#keras.utils.to_categorical(X,num_classes = 10)
#keras.utils.to_categorical(y,num_classes = 10)
X/255.0


# In[78]:




X.shape


# In[101]:


model = Sequential()

#model.add(Conv2D(32, (3,3),data_format="channels_last", input_shape=(50,50,3)))
#model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation='relu', input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(16, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(GlobalAveragePooling2D())
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


# In[102]:


NAME = "CNN"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
model.fit(X,y,batch_size=32,epochs=10, validation_split=0.4,callbacks=[tensorboard])


# In[88]:


model.save('material_detection.model')


# In[98]:


import cv2
CATEGORIES = ["Dog", "Cat"]
def prepare(filepath):
    IMG_SIZE = 50  
    img_array = cv2.imread(filepath, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


# In[99]:


model = tf.keras.models.load_model("material_detection.model")
prediction = model.predict([prepare('tree.jpg')])
print(prediction)


# In[100]:


print(CATEGORIES[int(prediction[0][0])])

