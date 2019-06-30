# catsDogs Data Augmentation example

# select 1000 images for training and 400 for validation from 12500 images

from PIL import Image
import glob
import os
# In the directory where you folder Cat and Dog are located, we create new folder
# data1000. 

os.chdir('./')

image_list_cat = []
image_list_dog = []

# CATS
for file in glob.glob('Cat/*.jpg'):
    # create a list of 1000 images
    while len(image_list_cat) < 1400:
        im = Image.open(file)
        image_list_cat.append(im)
        break   
# add to training set of cat images
for i, image in enumerate(image_list_cat[:1000]):
    image.convert('RGB').save('data1000/train/cats/image_' + str(i) + '.jpg')

# add to testing set of cat images
for i, image in enumerate(image_list_cat[1000:]):
    image.convert('RGB').save('data1000/test/cats/image_' + str(i) + '.jpg')

# DOGS
for file in glob.glob('Dog/*.jpg'):
    # create a list of 1000 images
    while len(image_list_dog) < 1400:
        im = Image.open(file)
        image_list_dog.append(im)
        break   
# add to training set of dog images
for i, image in enumerate(image_list_dog[:1000]):
    image.convert('RGB').save('data1000/train/dogs/image_' + str(i) + '.jpg')

# add to testing set of dog images
for i, image in enumerate(image_list_dog[1000:]):
    image.convert('RGB').save('data1000/test/dogs/image_' + str(i) + '.jpg')


