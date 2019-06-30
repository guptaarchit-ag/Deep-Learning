from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

os.chdir('.\data1000')

datadir = '.\data1000'

img = load_img('Cat/3.jpg')  
# this is a PIL image
x = img_to_array(img)  
x.shape
x = x.reshape((1,) + x.shape)  
 
# Build model architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

# Model 3 - same architecture for color images with data augmentation
model3 = Sequential()

model3.add(Conv2D(64,(3,3),input_shape=(50,50,3)))
model3.add(Activation('relu'))
model3.add(MaxPooling2D(pool_size = (2,2)))

model3.add(Conv2D(64,(3,3)))
model3.add(Activation('relu'))
model3.add(MaxPooling2D(pool_size = (2,2)))

model3.add(Flatten())
model3.add(Dense(64)) 

model3.add(Dense(1))
model3.add(Activation('sigmoid'))

model3.compile(loss= 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

# We use .flow_from_directory() to generate batches of image data directly from our jpgs 
# in their respective folders.

batch_size = 16
# Use the following data augmentation configuation
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range=0.2,
        zoom_range =0.2,
        horizontal_flip = True)

# For testing we use the following augmentation
test_datagen = ImageDataGenerator(rescale = 1./255)

# The following is generator that will read pictures found in the subfolders 
# 'data1000/train' and indefinitely generate batches of augmented image
# data
train_generator = train_datagen.flow_from_directory(
        "data1000/train",
        target_size = (50,50),
        batch_size = batch_size,
        class_mode = "binary")

# similar generator for validation data
validation_generator = test_datagen.flow_from_directory(
        "data1000/test",
        target_size = (50,50),
        batch_size = batch_size,
        class_mode= "binary")

# we now use the generator to train our model
model3.fit_generator(
        train_generator,
        steps_per_epoch = 2000//batch_size,
        epochs = 10,
        validation_data = validation_generator,
        validation_steps = 8000//batch_size)
model3.save_weights('first_try.h5') 

