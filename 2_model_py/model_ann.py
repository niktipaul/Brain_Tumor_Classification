import pickle
from keras import models, layers
import matplotlib.pyplot as plt


# Importing the Numpy Arrays
print('\n\n\n\n======== ANN Training Starts Here ========\n')


fh_train = open('4_data\\numpy_train.dat','rb+')
fh_test = open('4_data\\numpy_test.dat','rb+')
fh_class = open('4_data\\classes.dat','rb+')

image_train = pickle.load(fh_train)
image_test = pickle.load(fh_test)
classes = pickle.load(fh_class)
class_train = classes[0]
class_test = classes[1]

# For ANN we need to convert it to 2D

print("[+]Converting Numpy Arrays to 2D")
image_train = image_train.reshape(len(image_train),-1)
image_test = image_test.reshape(len(image_test),-1)
print("[+]Image Converted Successfully!!!")

# Building our ANN Model

print("\n[+]Training ANN Model Started...")
ann = models.Sequential([

    layers.Dense(360,input_shape = (224*224,),activation='sigmoid'),
    layers.Dropout(0.2),

    layers.Dense(120,activation='sigmoid'),
    layers.Dropout(0.1),

    layers.Dense(30,activation='sigmoid'),
    layers.Dropout(0.1),

    layers.Dense(2,activation='softmax')
])

# Compiling the Model

print("\n[+]Compiling ANN Model Started...")
ann.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
saved_history = ann.fit(image_train,class_train, validation_data = (image_test,class_test), epochs = 10)
print("[+]Training Finished Successfully!!!\n")


# Saving the Trained ANN  model

print("\n[+]Saving the ANN Neighbour in ann.json and weight in ann.h5...")
ann_json = ann.to_json()
with open("5_models\\ann.json", "w+") as json_file:
    json_file.write(ann_json)
ann.save_weights("5_models\\ann.h5")

# Saving the Accuracy of ANN model

print('\n[+]Saving the Accuracy Curve...')
fig1 = plt.figure("Figure 1")
plt.plot(saved_history.history['accuracy'])
plt.plot(saved_history.history['val_accuracy'])
plt.title('Model Accuracy Curve')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['accuracy','val_accuracy'],loc = 'lower right')
plt.savefig("7_saved_curves\\ann_accuracy_train.jpg")

# Saving the Loss of ANN model

print('[+]Saving the Loss Curve...')
fig2 = plt.figure("Figure 2")
plt.plot(saved_history.history['loss'])
plt.plot(saved_history.history['val_loss'])
plt.title('Model Loss Curve')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['loss','val_loss'], loc = 'upper right')
plt.savefig("7_saved_curves\\ann_loss_train.jpg")


print("[+]Files saved successfully!!!")

