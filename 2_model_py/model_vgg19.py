import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten



# Importing the Numpy Arrays
print('\n\n\n\n======== vgg19 Training Starts Here ========\n')


fh_train = open('4_data\\numpy_train.dat','rb+')
fh_test = open('4_data\\numpy_test.dat','rb+')
fh_class = open('4_data\\classes.dat','rb+')

image_train = pickle.load(fh_train)
image_test = pickle.load(fh_test)
classes = pickle.load(fh_class)
class_train = classes[0]
class_test = classes[1]


# For vgg19 we need to convert it to 4D

print("[+]Converting Numpy Arrays to 4D")
image_train = image_train.reshape(len(image_train),240,240,1)
image_test = image_test.reshape(len(image_test),240,240,1)
print("[+]Image Converted Successfully!!!")


# Building our vgg19 Model

print("\n[+]Training vgg19 Model Started...")
vgg19 = Sequential([

    Conv2D(filters= 64, kernel_size = (3,3),input_shape=(240, 240, 1)),
    Conv2D(filters= 64, kernel_size = (3,3)),
    MaxPooling2D((3,3)),

    Conv2D(filters= 128, kernel_size = (3,3)),
    Conv2D(filters= 128, kernel_size = (3,3)),
    MaxPooling2D((3,3)),

    Conv2D(filters= 256, kernel_size = (3,3)),
    Conv2D(filters= 256, kernel_size = (3,3), activation= 'relu'),
    Conv2D(filters= 256, kernel_size = (3,3), activation= 'relu'),
    Conv2D(filters= 256, kernel_size = (3,3), activation= 'relu'),
    MaxPooling2D((3,3)),

    Conv2D(filters= 512, kernel_size = (3,3), activation= 'relu'),
    Conv2D(filters= 512, kernel_size = (3,3), activation= 'relu'),
    Conv2D(filters= 512, kernel_size = (3,3), activation= 'relu'),
    Conv2D(filters= 512, kernel_size = (3,3), activation= 'relu'),
    MaxPooling2D((3,3)),

    Conv2D(filters= 512, kernel_size = (3,3), activation= 'relu'),
    Conv2D(filters= 512, kernel_size = (3,3), activation= 'relu'),
    Conv2D(filters= 512, kernel_size = (3,3), activation= 'relu'),
    Conv2D(filters= 512, kernel_size = (3,3), activation= 'relu'),


    Flatten(),
    Dense(4096),
    Dense(4096),
    Dense(1000),
    Dense(2,activation='softmax')
])

# Compiling the Model

print("\n[+]Compiling vgg19 Model Started...")
vgg19.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
saved_history = vgg19.fit(image_train,class_train, validation_data = (image_test,class_test), epochs = 50 )
print("[+]Training Finished Successfully!!!\n")


# Saving the Trained vgg19 Neighbour model

print("\n[+]Saving the vgg19 Neighbour in vgg19.json and weight in vgg19.h5...")
vgg19_json = vgg19.to_json()
with open("5_models\\vgg19.json", "w+") as json_file:
    json_file.write(vgg19_json)
vgg19.save_weights("5_models\\vgg19.h5")
print("[+]File saved successfully!!!")

# Saving the Accuracy of vgg19 model

print('\n[+]Saving the Accuracy Curve...')
fig1 = plt.figure("Figure 1")
plt.plot(saved_history.history['accuracy'])
plt.plot(saved_history.history['val_accuracy'])
plt.title('Model Accuracy Curve')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['accuracy','val_accuracy'],loc = 'lower right')
plt.savefig("7_saved_curves\\vgg19_accuracy_train.jpg")

# Saving the Loss of vgg19 model

print('[+]Saving the Loss Curve...')
fig2 = plt.figure("Figure 2")
plt.plot(saved_history.history['loss'])
plt.plot(saved_history.history['val_loss'])
plt.title('Model Loss Curve')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['loss','val_loss'], loc = 'upper right')
plt.savefig("7_saved_curves\\vgg19_loss_train.jpg")


print("[+]Files saved successfully!!!")