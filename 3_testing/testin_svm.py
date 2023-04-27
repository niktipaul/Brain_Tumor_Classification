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


print('\n\n\n\n\n======== Testing SVM Starts Here ========\n')

# Importing our SV Model

fh_svm = open('5_models\\svm.sav','rb+')
supportVector = pickle.load(fh_svm)

# Importing Test Data set

print("[+]Importing the dataset...")

# Assigning two list to store images and thier respective class

image_set = []
class_set = []

for cls in classes:
  pth = image_path + cls

  for current_image in os.listdir(pth):
    if current_image.split('.')[1] == 'jpg':
        img = cv2.imread(pth + '\\' + current_image,0)
        img = cv2.resize(img,(224,224))
        image_set.append(np.array(img))
        class_set.append(classes[cls])
print("[+]Importing Successfull!!!")


# Normalizing the image sets

print("[+]Normalizing the Train and Validation Image Sets...")
image_set = normalize(image_set, axis=1)
print("[+]Normalizing successfull!!!")


# Predicting Images:

print("\n[+]Beginning Prediction For SVM...")

image_set = image_set.reshape(len(image_set),-1)
predicted_set = supportVector.predict(image_set)

print("[+]Prediction Finished!!!\n")
print("\n===== OVERALL EVALUATION DETAILS FOR SV MODEL =====\n")

# Total Images

total_images = len(predicted_set)
print("Total Images",total_images)

# Calculating Accuracy

accuracy_var = round(accuracy_score(class_set,predicted_set)*100,2)
print("\nAccuracy of SVM Model:",accuracy_var)


# Calculating Precision

precision = round(precision_score(class_set,predicted_set)*100,2)
print("Precision of SVM:",precision)

# Calculating Recall

recall = round(recall_score(class_set,predicted_set)*100,2)
print("Recall of SVM:",recall)

# Calculating F1 Score

f1 = round(f1_score(class_set,predicted_set)*100,2)
print("F1 Score of SVM:",f1)

# Custom Labels
current_label = supportVector.classes_
class_labels = {0: 'No Tumor', 1: 'Glioma Tumor'}
custom_labels = []
for label in current_label:
  custom_labels.append(class_labels[label])

# For Confusion Matrix
cm = confusion_matrix(class_set,predicted_set)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = custom_labels)
cm_display.plot()
plt.gcf().set_size_inches(7, 5)
plt.savefig("6_confusion_matrix\\svm_confMat.jpg")

# Updating STATS
csv_path = "8_statistics\\stats.csv"
my_funcs.write_as_stats_u('SVM',csv_path,accuracy_var,precision,recall,f1,total_images)
print("\n[+]Stats updated successfully!!!")

