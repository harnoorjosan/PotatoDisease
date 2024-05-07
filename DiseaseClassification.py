#!/usr/bin/env python
# coding: utf-8

# In[24]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np


# In[25]:


BASE_DIR = "PlantVillage"
IMG_SIZE = 256 # check image proprties
BATCH_SIZE = 32 # standard convention 
EPOCH = 50 # at end of every epoch, we do validation
CHANNELS = 3
dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=BASE_DIR, 
        shuffle=True, 
        image_size=(IMG_SIZE, IMG_SIZE), 
        batch_size=BATCH_SIZE
    )


# In[26]:


class_names = dataset.class_names
class_names


# In[27]:


'''
Total images = 2152
Batch size = 32 
Total batches = 2152/32 = 68 = length of dataset
80% training data = 54 batches
10% validation data = 6 batches
10% testing data = 8 batches
'''
train_ds = dataset.take(54) #80% data is training
len(train_ds)


# In[28]:


non_train_ds = dataset.skip(54)
val_ds = non_train_ds.take(6)
len(val_ds)


# In[29]:


test_ds = non_train_ds.skip(6)
len(test_ds)


# In[30]:


def get_ds_partitions(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split*ds_size)
    train_ds = ds.take(train_size)
    
    val_size = int(val_split*ds_size)
    val_ds = ds.skip(train_size).take(val_size)
    
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


# In[31]:


train_ds, val_ds, test_ds = get_ds_partitions(dataset)


# In[32]:


'''
prefetch = CPU starts reading batch 2 even before GPU is done training on batch1.
cache = Not reading the same image again in epoch2 if it has already been read once in epoch1. 
'''
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[33]:


'''
Preprocessing
1. Rescale: The image pixel varies from a number 0 to 255. We need to divide each pixel by 255, such that they are between 0 and 1 only.
2. Resize: Similarly, we need to ensure all images are 256x256. Therefore, rescale the dataset. Although all our images are 256x256 but we still need to ensure.
3. Data augmentation: We create more data by rotating, flipping, changing contrast of existing data so our modls performs better.
'''
resize_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1.0/255)
])
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])


# In[34]:


# CNN layers: Trial and eroor of alternating Conv2D and MaxPooling
n_classes = 3
input_shape = (BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANNELS)
model = models.Sequential([
    resize_rescale,
    data_augmentation,
    layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(32,256,256,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')    
])
model.build(input_shape=input_shape)


# In[35]:


model.summary()


# In[36]:


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics = ['accuracy'])


# In[37]:


history = model.fit(train_ds, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1, validation_data=val_ds)


# In[39]:


scores = model.evaluate(test_ds)


# In[40]:


scores


# In[41]:


history.history.keys()


# In[43]:


import numpy as np
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0) # create a batch
    predictions=model.predict(img_array)
    predicted_class=class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])), 2)
    return predicted_class, confidence


# In[48]:


plt.figure(figsize = (15,15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]
        plt.title(f"Actual: {actual_class} \n Predicted: {predicted_class} \n Confidence: {confidence}")
        plt.axis("off")


# In[ ]:





# In[ ]:




