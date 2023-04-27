# FOR vgg19

import cv2
import os
import numpy as np
from keras.utils import normalize
from keras.models import model_from_json
import my_funcs
from global_vars import image_path, classes
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



print('\n\n\n\n\n======== Testing vgg19 Starts Here ========\n')

# Importing our vgg19 Model
fh_vgg19 = open('5_models\\vgg19.json','r+')
vgg19_json = fh_vgg19.read()
fh_vgg19.close()
vgg19 = model_from_json(vgg19_json)
vgg19.load_weights("5_models\\vgg19.h5")


# Importing Test Data set

print("[+]Importing the dataset...")

# Assigning two list to store images and thier respective class

image_set = []
class_set = []

for cls in classes:
  pth = image_path + cls

  for current_image in os.listdir(pth):
      img = cv2.imread(pth + '\\' + current_image,0)
      img = cv2.resize(img,(240,240))
      image_set.append(np.array(img))
      class_set.append(classes[cls])
print("[+]Importing Successfull!!!")


# Normalizing the image sets

print("[+]Normalizing the Train and Validation Image Sets...")
image_set = normalize(image_set, axis=1)
print("[+]Normalizing successfull!!!")

# Predicting Images:

print("\n[+]Beginning Prediction For vgg19...")

predicted = vgg19.predict(image_set)
predicted_set = []
for arr in predicted:
  predicted_set.append(np.argmax(arr))

print("[+]Prediction Finished!!!\n")
print("\n===== OVERALL EVALUATION DETAILS FOR vgg19 MODEL =====\n")

# Total Images

total_images = len(predicted_set)
print("Total Images",total_images)

# Calculating Accuracy

accuracy_var = round(accuracy_score(class_set,predicted_set)*100,2)
print("\nAccuracy of vgg19 Model =",accuracy_var)

# Calculating Precision

precision = round(precision_score(class_set,predicted_set)*100,2)
print("Precision of vgg19:",precision)

# Calculating Recall

recall = round(recall_score(class_set,predicted_set)*100,2)
print("Recall of vgg19:",recall)

# Calculating F1 Score

f1 = round(f1_score(class_set,predicted_set)*100,2)
print("F1 Score of vgg19:",f1)

# Custom Labels
current_label = [0,1]
class_labels = {0: 'No Tumor', 1: 'Glioma Tumor'}
custom_labels = []
for label in current_label:
  custom_labels.append(class_labels[label])

# For Confusion Matrix
cm = confusion_matrix(class_set,predicted_set)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = custom_labels)
cm_display.plot()
plt.gcf().set_size_inches(7, 5)
plt.savefig("6_confusion_matrix\\vgg19_confMat.jpg")

# Updating STATS

csv_path = "8_statistics\\stats.csv"
my_funcs.write_as_stats_u('vgg19',csv_path,accuracy_var,precision,recall,f1,total_images)
print("\n[+]Stats updated successfully!!!")

