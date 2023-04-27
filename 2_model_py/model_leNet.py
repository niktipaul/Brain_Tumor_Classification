import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, AveragePooling2D, Flatten, BatchNormalization



# Importing the Numpy Arrays
print('\n\n\n\n======== leNet Training Starts Here ========\n')


fh_train = open('4_data\\numpy_train.dat','rb+')
fh_test = open('4_data\\numpy_test.dat','rb+')
fh_class = open('4_data\\classes.dat','rb+')

image_train = pickle.load(fh_train)
image_test = pickle.load(fh_test)
classes = pickle.load(fh_class)
class_train = classes[0]
class_test = classes[1]


# For leNet we need to convert it to 4D

print("[+]Converting Numpy Arrays to 4D")
image_train = image_train.reshape(len(image_train),224,224,1)
image_test = image_test.reshape(len(image_test),224,224,1)
print("[+]Image Converted Successfully!!!")


# Building our leNet Model

print("\n[+]Training leNet Model Started...")
leNet = Sequential([


    Conv2D(filters=6, kernel_size=(3, 3), activation='relu'),
    AveragePooling2D(),

    Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    AveragePooling2D(),


    Flatten(),
    Dense(120,activation='relu'),
    Dense(84,activation='relu'),
    Dense(10,activation='softmax')  
])

# Compiling the Model

print("\n[+]Compiling leNet Model Started...")
leNet.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
saved_history = leNet.fit(image_train,class_train, validation_data = (image_test,class_test), epochs = 10 )
print("[+]Training Finished Successfully!!!\n")


# Saving the Trained leNet Neighbour model

print("\n[+]Saving the leNet Neighbour in leNet.json and weight in leNet.h5...")
leNet_json = leNet.to_json()
with open("5_models\\leNet.json", "w+") as json_file:
    json_file.write(leNet_json)
leNet.save_weights("5_models\\leNet.h5")
print("[+]File saved successfully!!!")

# Saving the Accuracy of leNet model

print('\n[+]Saving the Accuracy Curve...')
fig1 = plt.figure("Figure 1")
plt.plot(saved_history.history['accuracy'])
plt.plot(saved_history.history['val_accuracy'])
plt.title('Model Accuracy Curve')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['accuracy','val_accuracy'],loc = 'lower right')
plt.savefig("7_saved_curves\\leNet_accuracy_train.jpg")

# Saving the Loss of leNet model

print('[+]Saving the Loss Curve...')
fig2 = plt.figure("Figure 2")
plt.plot(saved_history.history['loss'])
plt.plot(saved_history.history['val_loss'])
plt.title('Model Loss Curve')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['loss','val_loss'], loc = 'upper right')
plt.savefig("7_saved_curves\\leNet_loss_train.jpg")


print("[+]Files saved successfully!!!")