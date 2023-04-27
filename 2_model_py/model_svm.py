from sklearn.svm import SVC
import pickle



# Importing the Numpy Arrays
print('\n======== SVM Training Starts Here ========\n')


fh_train = open('4_data\\numpy_train.dat','rb+')
fh_test = open('4_data\\numpy_test.dat','rb+')
fh_class = open('4_data\\classes.dat','rb+')

image_train = pickle.load(fh_train)
image_test = pickle.load(fh_test)
classes = pickle.load(fh_class)
class_train = classes[0]
class_test = classes[1]

# For SVM we need to convert it to 2D

print("[+]Converting Numpy Arrays to 2D")
image_train = image_train.reshape(len(image_train),-1)
image_test = image_test.reshape(len(image_test),-1)
print("[+]Image Converted Successfully!!!")

# Training Support Vector Model

print("[+]Training SV Model...")
supportVector = SVC(kernel='rbf', tol = 0.001)
supportVector.fit(image_train, class_train)
print("[+]Training Finished Successfully!!!\n")

# Saving the Trained Support vector model

print("\n[+]Saving the Support Vector Model in svm.sav")
fh_svm = open('5_models\\svm.sav','wb+')
pickle.dump(supportVector,fh_svm)
print("[+]File saved successfully!!!")
