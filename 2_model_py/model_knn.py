from sklearn.neighbors import KNeighborsClassifier
import pickle


# Importing the Numpy Arrays
print('\n======== KNN Training Starts Here ========\n')


fh_train = open('4_data\\numpy_train.dat','rb+')
fh_test = open('4_data\\numpy_test.dat','rb+')
fh_class = open('4_data\\classes.dat','rb+')

image_train = pickle.load(fh_train)
image_test = pickle.load(fh_test)
classes = pickle.load(fh_class)
class_train = classes[0]
class_test = classes[1]

# For KNN we need to convert it to 2D

print("[+]Converting Numpy Arrays to 2D")
image_train = image_train.reshape(len(image_train),-1)
image_test = image_test.reshape(len(image_test),-1)
print("[+]Image Converted Successfully!!!")

# Training K-Nearest Neighbour Model

print("[+]Training K-Nearest Neighbour Model...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(image_train, class_train)
print("[+]Training Finished Successfully!!!\n")

# Saving the Trained K-Nearest Neighbour model

print("\n[+]Saving the K-Nearest Neighbour in knn.sav")
fh_knn = open('5_models\\knn.sav','wb+')
pickle.dump(knn,fh_knn)
print("[+]File saved successfully!!!")
