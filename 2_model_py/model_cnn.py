import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten



# Importing the Numpy Arrays
print('\n\n\n\n======== CNN Training Starts Here ========\n')


fh_train = open('4_data\\numpy_train.dat','rb+')
fh_test = open('4_data\\numpy_test.dat','rb+')
fh_class = open('4_data\\classes.dat','rb+')

image_train = pickle.load(fh_train)
image_test = pickle.load(fh_test)
classes = pickle.load(fh_class)
class_train = classes[0]
class_test = classes[1]


# For CNN we need to convert it to 4D

print("[+]Converting Numpy Arrays to 4D")
image_train = image_train.reshape(len(image_train),224,224,1)
image_test = image_test.reshape(len(image_test),224,224,1)
print("[+]Image Converted Successfully!!!")


# Building our CNN Model

print("\n[+]Training CNN Model Started...")
cnn = Sequential([

    Conv2D(filters= 128, kernel_size = (3,3), activation= 'relu'),
    Dropout(0.25),
    MaxPooling2D((2,2)),

    Conv2D(filters= 64, kernel_size = (3,3), activation= 'relu'),
    Dropout(0.25),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(32,activation='sigmoid'),
    Dropout(0.1),
    Dense(2,activation='softmax')
])

# Compiling the Model

print("\n[+]Compiling CNN Model Started...")
cnn.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
saved_history = cnn.fit(image_train,class_train, validation_data = (image_test,class_test), epochs = 10 )
print("[+]Training Finished Successfully!!!\n")
 
 
# Saving the Trained CNN Neighbour model

print("\n[+]Saving the CNN in cnn.json and weight in cnn.h5...")
cnn_json = cnn.to_json()
with open("5_models\\cnn.json", "w+") as json_file:
    json_file.write(cnn_json)
cnn.save_weights("5_models\\cnn.h5")
print("[+]File saved successfully!!!")

# Saving the Accuracy of CNN model

print('\n[+]Saving the Accuracy Curve...')
fig1 = plt.figure("Figure 1")
plt.plot(saved_history.history['accuracy'])
plt.plot(saved_history.history['val_accuracy'])
plt.title('Model Accuracy Curve')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['accuracy','val_accuracy'],loc = 'lower right')
plt.savefig("7_saved_curves\\cnn_accuracy_train.jpg")

# Saving the Loss of CNN model

print('[+]Saving the Loss Curve...')
fig2 = plt.figure("Figure 2")
plt.plot(saved_history.history['loss'])
plt.plot(saved_history.history['val_loss'])
plt.title('Model Loss Curve')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['loss','val_loss'], loc = 'upper right')
plt.savefig("7_saved_curves\\cnn_loss_train.jpg")


print("[+]Files saved successfully!!!")