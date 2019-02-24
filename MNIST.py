#MNIST CNN Modelling
#Basic libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import itertools

#Model libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Import Keras library
from keras.utils.np_utils import to_categorical #For one hot encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, History
from keras.models import load_model
from keras.utils import plot_model

#Set Working directory
os.chdir('C:\\Users\\Santosh Selvaraj\\Documents\\R Working Directory\\Data Science Projects\\MNIST')

#Load Data
trainData = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")

#Create Dependents and Independents
ytrain = trainData['label']
Xtrain = trainData.drop(labels = "label", axis = 1)

#Free memory
del trainData

#Data Exploration
#sns.countplot(ytrain) # Count plot for ytrain
#ytrain.value_counts() # Value of counts in ytrain
#Xtrain.isnull().any().describe() # No nulls; One unique value FALSE
#testData.isnull().any().describe() # No nulls; One unique value FALSE

#Normalize the data
Xtrain = Xtrain/255.0
testData = testData/255.0

#Reshape the data as 3D tensor
Xtrain = Xtrain.values.reshape(42000,28,28,1)
testData = testData.values.reshape(28000,28,28,1)

#Plot the data and check
#plt.imshow(Xtrain[3,:,:,0], cmap="gray")

#Label Encoding
ytrain = to_categorical(ytrain, num_classes = 10)

#Split into training and test set
Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size = 0.1, random_state = 2)

#Build the CNN Architecture
#Conv2D-MaxPool2D-Dropout-Conv2D-MaxPool2D-Dropout-Flatten-Dense-Dropout-Output
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5),padding="same",activation="relu",input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units = 128, activation = "relu"))
model.add(Dropout(rate = 0.25))
model.add(Dense(units = 10, activation = "softmax"))

#Compiling the CNN
optimizer = RMSprop(lr = 0.001, rho = 0.9)
model.compile(optimizer=optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

#Dynamically reduce the Learning Rate if accuracy stagnates
learningRateReduction = ReduceLROnPlateau(monitor="val_loss",patience=3,verbose=1,factor=0.5,min_lr=0.00001)
history = History()
epochs = 10 #30
batch_size = 50 #86

#Compile the model without data augmentation
finalModel = model.fit(Xtrain,ytrain,batch_size=batch_size,epochs=epochs,validation_data=(Xval,yval),verbose=1,callbacks=[history,learningRateReduction])
#steps_per_epoch=Xtrain.shape[0]//batch_size
#callbacks=learningRateReduction

# With data augmentation to prevent overfitting (accuracy 0.99286)
#datagen = ImageDataGenerator(
#        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
#        zoom_range = 0.1, # Randomly zoom image 
#        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#        height_shift_range=0.1)  # randomly shift images vertically (fraction of total height)

#datagen.fit(Xtrain)

#finalModel = model.fit_generator(datagen.flow(Xtrain,ytrain, batch_size=batch_size),
#                              epochs = epochs, validation_data = (Xval,yval),
#                              verbose = 1, steps_per_epoch=Xtrain.shape[0] // batch_size
#                              , callbacks=[learningRateReduction,history])
#Save the model
model.save("CNNModel.h5")
#model = load_model("CNNModel.h5")

#Visualizations
# Plot training & validation accuracy values
plt.plot(finalModel.history['acc'])
plt.plot(finalModel.history['val_acc'])
plt.ylim(ymin = 0.9, ymax = 1)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

# Plot training & validation loss values
plt.plot(finalModel.history['loss'])
plt.plot(finalModel.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

#Confusion Matrix
#Get the predictions on validation set
Ypred = model.predict(Xval)
#Plot the first image in X validation
#plt.imshow(Xval[0,:,:,0], cmap="gray")
#Predict the classes as a single vector
Ypred = np.argmax(Ypred, axis = 1)
#The true classes as a vector
Ytrue = np.argmax(yval, axis = 1)
#Build the confusion matrix
cm = confusion_matrix(Ytrue, Ypred)

#Plot the confusion matrix
plt.imshow(cm,interpolation="nearest",cmap=plt.cm.Greens)
plt.title("Confusion Matrix")
tick_marks = np.arange(10)
plt.xticks(tick_marks)
plt.yticks(tick_marks)
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if i==j else "black")

#Finding the images that were not correctly predicted
PredErrors = Ytrue - Ypred != 0
XvalErrors = Xval[PredErrors]
Ytrueerrors = Ytrue[PredErrors]
Yprederrors = Ypred[PredErrors]

#Check Errors
def plot_errors(n=0,nrows=2,ncols=3):
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            ax[row,col].imshow((XvalErrors[n,:,:,0]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(Yprederrors[n],Ytrueerrors[n]))
            n += 1

#plot_errors(20)

#Final Predictions on Test Data
results = model.predict(testData)
#Select the predictions with max probability
results = np.argmax(results, axis = 1)
results = pd.Series(results, name = "Label")

#Creating the csv
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("mnistCNNPreds.csv",index=False)