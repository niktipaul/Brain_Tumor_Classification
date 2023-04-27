# FOR SUPPORT VECTORE MACHINE

import cv2
import os
import numpy as np
import pickle
from keras.utils import normalize
import my_funcs
from global_vars import image_path, classes
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import pickle
import matplotlib.pyplot as plt
from sklearn import svm
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization, Softmax, ZeroPadding2D


print('\n\n\n\n\n======== Testing CNN SVM Starts Here ========\n')

# Importing our SV Model

fh_svm = open('5_models\\svm_cnn_latif.sav','rb+')
supportVector = pickle.load(fh_svm)

# Importing Test Data set

print("[+]Importing the dataset...")

# Assigning two list to store images and thier respective class

test_image_set = []
test_class_set = []

for cls in classes:
  pth = image_path + cls

  for current_image in os.listdir(pth):
    if current_image.split('.')[1] == 'jpg':
        img = cv2.imread(pth + '\\' + current_image,0)
        img = cv2.resize(img,(241,241))
        img = cv2.merge((img,img,img))
        test_image_set.append(np.array(img))
        test_class_set.append(classes[cls])
print("[+]Importing Successfull!!!\n")

# Converting to Numpy Array


print("[+]Converting to Numpy Array...")
test_image_set = np.array(test_image_set)
test_class_set = np.array(test_class_set)
print("[+]Images converted to Numpy Array Successfully!!!\n")


# Normalizing the image sets

print("[+]Normalizing the Test Data Sets...")
test_image_set = normalize(test_image_set, axis=1)
print("[+]Normalizing successfull!!!\n")

# Into CNN Architecture:

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

# Extracting featrures from test data:

print('[+]Extracting Features for train and test data...')
test_img_data_for_svm = feature_extractor.predict(test_image_set)
print("\n[+]Successfull!!!\n")


# Predicting Images:
print("[+]Predicting Result...")
predicted_set = supportVector.predict(test_img_data_for_svm)
print("\n[+]Successfull!!!\n")



print('[+]OVERALL PERFORMANCE OF THE MODEL...')
# Total Images
total_images = len(predicted_set)
print("Total Images",total_images)

# Calculating Accuracy
accuracy_var = round(accuracy_score(test_class_set,predicted_set)*100,2)
print("\nAccuracy of SVM Model:",accuracy_var)

# Calculating Precision
precision = round(precision_score(test_class_set,predicted_set)*100,2)
print("Precision of SVM:",precision)

# Calculating Recall
recall = round(recall_score(test_class_set,predicted_set)*100,2)
print("Recall of SVM:",recall)

# Calculating F1 Score
f1 = round(f1_score(test_class_set,predicted_set)*100,2)
print("F1 Score of SVM:",f1)
print('[+]Successfull!!\n')


print('Building Confusion Matrix...')
# Custom Labels
current_label = supportVector.classes_
class_labels = {0: 'No Tumor', 1: 'Glioma Tumor'}
custom_labels = []
for label in current_label:
  custom_labels.append(class_labels[label])

# For Confusion Matrix
cm = confusion_matrix(test_class_set,predicted_set)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = custom_labels)
cm_display.plot()
plt.gcf().set_size_inches(7, 5)
plt.savefig("6_confusion_matrix\\latif_confMat.jpg")
print('Successfull!!!\n')


print('[+]Updating Stat...')
# Updating STATS
csv_path = "8_statistics\\Stats.csv"
my_funcs.write_as_stats_u('CNN_SVM',csv_path,accuracy_var,precision,recall,f1,total_images)
print("[+]Stats updated successfully!!!\n")

