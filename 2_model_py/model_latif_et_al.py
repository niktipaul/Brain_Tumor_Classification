# This is a simulation of Latif et al's CNN SVM Architecture

import pickle
import matplotlib.pyplot as plt
from sklearn import svm
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization, Softmax, ZeroPadding2D

from sklearn.svm import SVC
import pickle

# Importing the Numpy Arrays
print('\n\n\n\n======== CNN SVM Training Starts Here ========\n')


fh_train = open('4_data\\numpy_train.dat','rb+')
fh_test = open('4_data\\numpy_test.dat','rb+')
fh_class = open('4_data\\classes.dat','rb+')

train_image_set = pickle.load(fh_train)  # image_train
image_test = pickle.load(fh_test) # NOT NEEDED (VALIDATION IMAGE SET)
classes = pickle.load(fh_class)

train_class_set = classes[0]  # class_train
class_test = classes[1]   # NOT NEEDED (VALIDATION CLASS SET)



# For CNN SVM we need to convert it to 4D

print("[+]Converting Numpy Arrays to 4D...")
train_image_set = train_image_set.reshape(len(train_image_set),241,241,3)
print("[+]Image Converted Successfully!!!\n")


# Building our CNN SVM Model

print("\n[+]Into CNN Architecture...")
feature_extractor = Sequential()

feature_extractor.add(Conv2D(filters= 96, kernel_size = (9,9), strides= (4,4), activation= 'relu'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(Softmax())
feature_extractor.add(MaxPooling2D((3,3), strides=(2,2)))
feature_extractor.add(ZeroPadding2D(padding=2))

feature_extractor.add(Conv2D(filters= 256, kernel_size = (7,7), strides= (1,1), activation= 'relu'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(Softmax())
feature_extractor.add(MaxPooling2D((3,3), strides=(2,2)))

feature_extractor.add(Conv2D(filters= 384, kernel_size = (3,3), strides= (1,1), activation= 'relu', padding='same'))

feature_extractor.add(Conv2D(filters= 384, kernel_size = (3,3), strides= (1,1), activation= 'relu', padding='same'))

feature_extractor.add(Conv2D(filters= 256, kernel_size = (3,3), strides= (1,1), activation= 'relu', padding='same'))
feature_extractor.add(Softmax())
feature_extractor.add(MaxPooling2D((3,3), strides=(2,2)))
feature_extractor.add(Flatten())
feature_extractor.add(Dense(4096, activation = 'softmax' ))

print("\n[+]Successfull!!!\n")



# Extracting Features
print('[+]Extracting Features for train data...')
train_img_data_for_svm = feature_extractor.predict(train_image_set)
print("\n[+]Successfull!!!\n")



print("[+]Training SV Model...")
supportVector = SVC(kernel='rbf', gamma= 0.0001, C= 1)
supportVector.fit(train_img_data_for_svm, train_class_set)
print("[+]Training Finished Successfully!!!\n")



print("\n[+]Saving the Support Vector Model in svm.sav")
fh_svm = open('5_models\\svm_cnn_latif.sav','wb+')
pickle.dump(supportVector,fh_svm)
print("[+]File saved successfully!!!\n")