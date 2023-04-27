# Brain_Tumor_Classification

The file contains 8 folders:
1_data_py
2_model_py
3_testing

4_data
5_models
6_confusion_matrix
7_saved_curves
8_statistics

1. "1_data_py" folder contains 3 python files: 
  a) dataset_pth.py  : Which contains the path and name of dataset
  b) loading_data.py  : This file loads the dataset converts the dataset images into Numpy Array, Normalizes the images, Splits the datasets into Train and validation sets and saves the datasets as binary files in "4_data" folder. Image ratio is 224 x 224.
  c) loading_data_v2.py  : This works same a "loading_data.py" but it is specially written for "Latif_et_al_CNN_SVM model". (241 x 241 image dimention).

2. "2_model_py" folder contains the python files of all the models. Each files load the binary format of train and validation datasets from the "4_data" folder. Trains the model and stores the trained model to "5_model" folder so that for testing purpose these pre-trained model can be used. 

3. "3_testing" folder contains the python files to test the pre-trained models which are stored in "5_model" folder. Each python file loads the test datasets and the pre-trained models. Then these pre-trained models are used to predict the class of test datasets. On the basis of result, Accuracy score, Precision, Recall and F1 Score are stored in "stats.csv" in the "8_statistics" folder. Also confusion matrix is generated and stored in "6_confusion_matrix" folder.
