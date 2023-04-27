import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, BatchNormalization



# Importing the Numpy Arrays
print('\n\n\n\n======== AlexNet Training Starts Here ========\n')


fh_train = open('4_data\\numpy_train.dat','rb+')
fh_test = open('4_data\\numpy_test.dat','rb+')
fh_class = open('4_data\\classes.dat','rb+')

image_train = pickle.load(fh_train)
image_test = pickle.load(fh_test)
classes = pickle.load(fh_class)
class_train = classes[0]
class_test = classes[1]


# For AlexNet we need to convert it to 4D

print("[+]Converting Numpy Arrays to 4D")
image_train = image_train.reshape(len(image_train),224,224,1)
image_test = image_test.reshape(len(image_test),224,224,1)
print("[+]Image Converted Successfully!!!")


# Building our AlexNet Model

print("\n[+]Training AlexNet Model Started...")
AlexNet = Sequential([


    Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D((3,3)),

    Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),

    Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),

    Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(1024,activation='relu'),
    Dropout(0.5),
    Dense(1024,activation='relu'),
    Dropout(0.5),
    Dense(10,activation='softmax')  
])

# Compiling the Model

print("\n[+]Compiling AlexNet Model Started...")
AlexNet.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
saved_history = AlexNet.fit(image_train,class_train, validation_data = (image_test,class_test), epochs = 10 )
print("[+]Training Finished Successfully!!!\n")


# Saving the Trained AlexNet model

print("\n[+]Saving the AlexNet in AlexNet.json and weight in AlexNet.h5...")
AlexNet_json = AlexNet.to_json()
with open("5_models\\AlexNet.json", "w+") as json_file:
    json_file.write(AlexNet_json)
AlexNet.save_weights("5_models\\AlexNet.h5")
print("[+]File saved successfully!!!")

# Saving the Accuracy of AlexNet model

print('\n[+]Saving the Accuracy Curve...')
fig1 = plt.figure("Figure 1")
plt.plot(saved_history.history['accuracy'])
plt.plot(saved_history.history['val_accuracy'])
plt.title('Model Accuracy Curve')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['accuracy','val_accuracy'],loc = 'lower right')
plt.savefig("7_saved_curves\\AlexNet_accuracy_train.jpg")

# Saving the Loss of AlexNet model

print('[+]Saving the Loss Curve...')
fig2 = plt.figure("Figure 2")
plt.plot(saved_history.history['loss'])
plt.plot(saved_history.history['val_loss'])
plt.title('Model Loss Curve')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['loss','val_loss'], loc = 'upper right')
plt.savefig("7_saved_curves\\AlexNet_loss_train.jpg")


print("[+]Files saved successfully!!!")