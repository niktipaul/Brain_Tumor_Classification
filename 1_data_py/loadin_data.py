import cv2
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from dataset_pth import classes, datasets, image_path


# And Importing the images
print('\n\n\n\n\n======== Data_Loading Starts Here ========+')

print("[+]Importing the dataset...")


# Assigning two list to store images and thier respective class

image_set = []
class_set = []


# IMPORTING Dataset 

for dset in datasets:
  for cls in classes:
    pth = image_path + dset + '\\training\\' + cls
    for current_image in os.listdir(pth):
        img = cv2.imread(pth + '\\' + current_image,0)
        img = cv2.resize(img,(224,224))
        image_set.append(np.array(img))
        class_set.append(classes[cls])
print("[+]Importing Successfull!!! ("+str(len(image_set))+" IMGS)")


# Converting the Datasets to numpy array

image_set = np.array(image_set)
class_set = np.array(class_set)
classes = []
print("[+]Images converted to Numpy Array Successfully!!!\n")

# Spliting Training set into TRAINING and VALIDATION

print("[+]Splitting data into Test and Validation...")
image_train,image_test,class_train,class_test = train_test_split(image_set,class_set, random_state = 1, test_size = 0.20)
classes.append(class_train)
classes.append(class_test)
print("[+]Splitting Successfull!!!\n")


# Normalizing the image sets (Feature Scaling)

print("[+]Normalizing the Train and Validation Image Sets...")
image_train = normalize(image_train, axis=1)
image_test = normalize(image_test, axis=1)
print("[+]Normalizing successfull!!!")




# Saving the Above Train and Validation datasets for future use

fh_train = open('4_data\\numpy_train.dat','wb+')
pickle.dump(image_train,fh_train)

fh_test = open('4_data\\numpy_test.dat','wb+')
pickle.dump(image_test,fh_test)

fh_class = open('4_data\\classes.dat','wb+')
pickle.dump(classes,fh_class)
print("[+]Numpy Data saved in File Successfully!!!")








