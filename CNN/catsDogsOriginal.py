import numpy as np 
import os
os.chdir('./data')

import cv2
import random
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

# Data preperation --- skip to part II if you have data saved as pickle file ----
datadir = './data' # you must have sub-directories in data called Cat and Dog 
                    # with the images 
categories = ['Dog','Cat']

img_size = 50
training_data = []

# prepare training data for grayscale images
def create_training_data():
    for category in categories:
        path = os.path.join(datadir,category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()

# training data for color images
training_data_rgb = []

def create_training_data_rgb():
    for category in categories:
        path = os.path.join(datadir,category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array_rgb = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
                new_array_rgb = cv2.resize(img_array_rgb,(img_size,img_size))
                training_data_rgb.append([new_array_rgb,class_num])
            except Exception as e:
                pass

create_training_data_rgb()

len(training_data)
len(training_data_rgb)

# Balancing the dataset 
random.shuffle(training_data)
random.shuffle(training_data_rgb)

for sample in training_data[:10]:
    print(sample[1])
    
X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,img_size,img_size,1)   

X_rgb = []
y_rgb = []
for features,label in training_data_rgb:
    X_rgb.append(features)
    y_rgb.append(label)

X_rgb = np.array(X_rgb).reshape(-1,img_size,img_size,3)   

pickle_out = open('X.pickle','wb')
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open('y.pickle','wb')
pickle.dump(y,pickle_out)
pickle_out.close()

pickle_out_rgb = open('X_rgb.pickle','wb')
pickle.dump(X_rgb,pickle_out_rgb)
pickle_out_rgb.close()

pickle_out_rgb = open('y_rgb.pickle','wb')
pickle.dump(y_rgb,pickle_out_rgb)
pickle_out_rgb.close()

# Once you have the data saved in pickle format, you can run the following 
# commmand to load it directly. 
# This will save time to perform basic operations discussed above.
pickle_in= open('X.pickle','rb')
X = pickle.load(pickle_in)
X[:3]

pickle_in2 = open('y.pickle','rb')
y = pickle.load(pickle_in2)
y[:10]

pickle_in_rgb = open('X_rgb.pickle','rb')
X_rgb  = pickle.load(pickle_in_rgb)

pickle_in2_rgb = open('y_rgb.pickle','rb')
y_rgb  = pickle.load(pickle_in2_rgb)

# standardize dataset
X = X/255.0
X.shape

X_rgb = X_rgb/255.0
X_rgb.shape

# Model1 - Grayscale images

model1 = Sequential()

model1.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size = (2,2)))

model1.add(Conv2D(64,(3,3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size = (2,2)))

model1.add(Flatten())
model1.add(Dense(64)) 

model1.add(Dense(1))
model1.add(Activation('sigmoid'))

model1.compile(loss= 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

model1.fit(X,y,batch_size = 32,validation_split = 0.1,epochs=10)

# With 10 epoch, batch size 32 and validation_split = 0.1, I got a validation 
# accuracy of 79.16%. 
