import warnings
warnings.filterwarnings('ignore')

# Required Moduls

from sklearn.linear_model import LogisticRegression
import pickle




# Importing the Numpy Arrays

print('\n======== Logistic Regression Training Starts Here ========\n')


fh_train = open('4_data\\numpy_train.dat','rb+')
fh_test = open('4_data\\numpy_test.dat','rb+')
fh_class = open('4_data\\classes.dat','rb+')

image_train = pickle.load(fh_train)
image_test = pickle.load(fh_test)
classes = pickle.load(fh_class)
class_train = classes[0]
class_test = classes[1]

# For Logistic Regression we need to convert it to 2D

print("[+]Converting Numpy Arrays to 2D")
image_train = image_train.reshape(len(image_train),-1)
image_test = image_test.reshape(len(image_test),-1)
print("[+]Image Converted Successfully!!!")

# Training Logistic Regression Model

print("[+]Training Logistic Regression Model...")
logReg = LogisticRegression(C = 0.1)
logReg.fit(image_train, class_train)
print("[+]Training Finished Successfully!!!\n")

# Saving the Trained Logistic Regression

print("\n[+]Saving the Logistic Regression in logr.sav")
fh_logr = open('5_models\\logr.sav','wb+')
pickle.dump(logReg,fh_logr)
print("[+]File saved successfully!!!")

