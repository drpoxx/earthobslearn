import glob
import os

import numpy as np
import pandas as pd
import seaborn as sns

from osgeo import gdal_array
from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split


#Keras packages to build Convolutional Neural Network architecture
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

def showConfMatrixCNN(model, X, y, label_str, norm=False):
    """
    Function to define the confusion matrix. This function uses the pre-trained model to predict the class of input matrix X.
    The predicted values (y_pred) are compared with the ground truth y and the final result show the errors of the classificator.

    Input: 
        model: Pretrained model.
        X: Array to be predicted.
        y: Ground truth.
        label_str: String that contains the name of the classes.
        norm: To see the normalized confusion matrixs. Defaults to False.

    Ouput:
        The Plot of the conufsion matrix.
    
    """
    
    print('[AI4EO_MOOC]_log: Prediction using pretrained model...')
    y_pred = model.predict(X)
    #prepare array
    y_pred_amax=np.zeros((len(y_pred)))    
    y_test_amax=np.zeros((len(y_pred)))
    #extract the class that contais the high value in prediction
    for i in range(0,len(y_pred)):
        y_pred_amax[i]=np.argmax(y_pred[i,:])
        y_test_amax[i]=np.argmax(y[i,:])
        
    print('[AI4EO_MOOC]_log: Create confusion matrix')
    matrix = metrics.confusion_matrix(y_test_amax, y_pred_amax)  
    if norm:
        matrix_norm = np.around(
                    matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis], decimals = 2)
        con_mat_df = pd.DataFrame(matrix_norm,
                                  index  = label_str,
                                  columns= label_str)
    
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            con_mat_df,
            annot=True,
            cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    else:
        con_mat_df = pd.DataFrame(matrix,
                                  index  = label_str,
                                  columns= label_str)
    
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            con_mat_df,
            annot=True,
            fmt='d',
            cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

if __name__ == "__main__":
    MAIN_PATH = 'C:\Users\ochsp\OneDrive\Personal projects\Data Science\Python dev\earthobslearn\earthobslearn'
    DATA_PATH = 'raw_data\tile_detection_training'

    #read data in a folder
    print('[AI4EO_MOOC]_log: Read all files in a main data folder')
    folder_for_training = glob.glob(MAIN_PATH+DATA_PATH+'*/')

    print('[AI4EO_MOOC]_log: There are %d folders' % (len(folder_for_training)))

    lst_arr_training=[]
    lst_gt_training =[]
    for i in range(0,len(folder_for_training)):
        data_for_training_tmp=glob.glob(folder_for_training[i]+'*.tif')
        
        print('[AI4EO_MOOC]_log: There are %d images for %s class' % (
            len(data_for_training_tmp), folder_for_training[i][116:-1])
            )
        
        for j in range(0,len(data_for_training_tmp)):
            arr_tmp = gdal_array.LoadFile(data_for_training_tmp[j])
            lst_arr_training.append(arr_tmp)
            tmp_gt = np.zeros(10) 
            tmp_gt[i]=1
            lst_gt_training.append(tmp_gt)

    print('[AI4EO_MOOC]_log: From list to multistack array')
    arr_training = np.asarray(lst_arr_training)
    #arr_gt = np.zeros((len(arr_training),1))
    arr_gt = np.asarray(lst_gt_training)

    num_of_img,bands,rows,columns=arr_training.shape
    print('[AI4EO_MOOC]_log: Reshape array from native shape (num_of_img:%d, bands:%d, rows:%d, columns:%d) to AI readble shape (num_of_img:%d, rows:%d, columns:%d, bands:%d). . .' % (
        num_of_img,bands,rows,columns, num_of_img,rows,columns,bands))
    arr_training_res = np.zeros((num_of_img,rows,columns,bands))
    for b in range(0,bands):
        arr_training_res[:,:,:,b] = arr_training[:,b,:,:]

    print('[AI4EO_MOOC]_log: Normalization data into [0,1] intervall...')
    arr_training_res = arr_training_res.astype('float32')
    for i in range(0, len(arr_training_res)):    
        amax_tmp=np.amax(arr_training_res[i,:,:,:])
        arr_training_res[i,:,:,:]/=amax_tmp

    test_size=0.15
    print('[AI4EO_MOOC]_log: Training (%0.2f %%) and validation (%0.2f %%) split..' % (
        (1-test_size)*100,(test_size)*100))
    X_train, X_test, y_train, y_test = train_test_split(
            arr_training_res,
            arr_gt, 
            test_size=test_size, 
            random_state=42)
    
    # CONVOLUTIONBAL MODEL.
    _,num_classes = y_train.shape
    print('[AI4EO_MOOC]_log: Convolutional Neural Network architecture:')
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model 
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    model.summary()

    save_dir = MAIN_PATH + 'trained_models/'
    model_name = 'keras_sentinel2_classification_trained_model_e50.json'
    print('[AI4EO_MOOC]_log: The model will be saved in the following path: %s' %(save_dir+model_name))

    #set path to save model with the best validation accuracy
    filepath_tmp = save_dir+model_name
    checkpoint = ModelCheckpoint(
        filepath_tmp, 
        'val_accuracy', 
        verbose=1, 
        save_best_only=True, 
        mode='max')
    callbacks_list = [checkpoint]

    # If data_augmentation is True, will be applied linear transformation to the input data
    data_augmentation = False
    batch_size = 32
    #Number of epochs
    epochs = 50

    print('[AI4EO_MOOC]_log: Selected parameters: ')
    print('[AI4EO_MOOC]_log: Batch size: %d' % (batch_size))
    print('[AI4EO_MOOC]_log: Num of epochs: %d' % (epochs))
    print('[AI4EO_MOOC]_log: Data Augmentation: %s' % (data_augmentation))

    if not data_augmentation:
        print('[AI4EO_MOOC]_log: Not using data augmentation.')    
        
        # Fit the model   
        print('[AI4EO_MOOC]_log: Fit the model with batch_size =', batch_size)    
        history = model.fit(X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks_list,
                verbose = 1)   
    
    else:
        print('[AI4EO_MOOC]_log: Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(X_train, y_train,
                                        batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            callbacks=callbacks_list,
                            workers=4)

    label_str=[
        'AnnualCrop',
        'Forest',
        'HerbaceousVegetation',
        'Highway',
        'Industrial',
        'Pasture',
        'PermanentCrop',
        'Residential',
        'River',
        'SeaLake' 
        ]
    norm=True

    showConfMatrixCNN(model, X_test, y_test, label_str,norm)