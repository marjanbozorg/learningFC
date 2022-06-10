import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.layers import ConvLSTM2D

dataframe = df = pd.read_csv('AirPassengers.csv', usecols=[1])
plt.plot(dataframe)

#convert pandas df to numpy array
dataset = dataframe.values
dataset = dataset.astype('float32')#convert values to float

#Nurmalization optional recomended for neural network
# in certain case the fun is sensitive to magnitude numbers

scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

#split data into train and test
#2/3 train and 1/3 test
train_size = int(len(dataset) * 0.66)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, : ], dataset[train_size:len(dataset),:]

def to_sequence(dataset, seq_size = 1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size -1):
        #print(i)
        window = dataset[i:(i+seq_size),0]
        x.append(window)
        y.append(dataset[i+seq_size,0])

    return np.array(x), np.array(y)

#num time steps to look back, larger sequences look further back  may improve FC
seq_size = 5

trainX, trainY = to_sequence(train, seq_size)
testX, testY = to_sequence(test, seq_size)

print("Shape of training set: {}".format(trainX.shape))
print("Shape of test set: {}".format(testX.shape))

#reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX,(trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX,(testX.shape[0], 1, testX.shape[1]))

print("Single LSTM with hidden Dense ...")

##############################################
#model = Sequential()
#model.add(LSTM(64, input_shape=(None, seq_size)))#12 too dense then overfitting
#only one model, it is optional
#model.add(Dense(32))#8
#model.add(Dense(1))#y value and next value
#model.compile(loss='mean_squared_error', optimizer='adam')
#monitor = EarlyStopping(monitor = 'vol_Loss', min_delta = 1e-3, patience = 20,
#                        verbose = 1, mode = 'auto', restore_best_weight = True)
#print(model.summary())
####################################################################

#reshape input to be [samples, time steps, features]
#trainX = np.reshape(trainX,(trainX.shape[0], 1, trainX.shape[1]))
#testX = np.reshape(testX,(testX.shape[0], 1, testX.shape[1]))
#model = Sequential()
#model.add(LSTM(50,activation= 'relu',return_sequences= True,input_shape=(None, seq_size)))#12 too dense then overfitting
#only one model, it is optional
#model.add(LSTM(50, activation='relu'))#8
#model.add(Dense(1))#y value and next value
#model.compile(loss='mean_squared_error', optimizer='adam')
#print(model.summary())
#####################################################################

#Bidirectional LSTM, let LSTM to learn in both forward and backward directions

#trainX = np.reshape(trainX,(trainX.shape[0], 1, trainX.shape[1]))
#testX = np.reshape(testX,(testX.shape[0], 1, testX.shape[1]))

#from keras.layers import Bidirectional
#model = Sequential()
#model.add(Bidirectional(LSTM(50,activation= 'relu'),input_shape=(None, seq_size)))#12 too dense then overfitting
#only one model, it is optional
#model.add(LSTM(50, activation='relu'))#8
#model.add(Dense(1))#y value and next value
#model.compile(loss='mean_squared_error', optimizer='adam')
#print(model.summary())

##############################################
#CovLSTM
#The layer expected in two dimentional images,
#the shape of input data must be: [samples, time steps, rows, columns,

trainX = np.reshape(trainX,(trainX.shape[0], 1,1,1, seq_size))
testX = np.reshape(testX,(testX.shape[0], 1,1,1, seq_size))

model = Sequential()
model.add(ConvLSTM2D(LSTM(filters= 64, kernel_size = (1,1),activation= 'relu',input_shape=(None, seq_size))))#12 too dense then overfitting
model.add(Flatten())
#only one model, it is optional
#model.add(LSTM(50, activation='relu'))#8
model.add(Dense(32))
model.add(Dense(1))#y value and next value
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())





model.fit(trainX, trainY,validation_data=(testX, testY),
          verbose=2, epochs=100)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


#since I have used minmaxscaler now use scaler.inverse_transform to invert the transformation
#do inverse transform
trainPredict = scaler.inverse_transform(trainPredict)
trainY_inverse = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY_inverse = scaler.inverse_transform([testY])

#calculate root mean square
trainScore = math.sqrt(mean_squared_error(trainY_inverse[0], trainPredict[:,0]))
print('Train Score:%.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY_inverse[0], testPredict[:,0]))
print('Test Score:%.2f RMSE' % (testScore))


#shift train prediction for plotting, aligne x axis with the orig one
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[seq_size:len(trainPredict)+seq_size,:]  = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(trainPredict)+(seq_size *2)+1:len(dataset)-1,:]  = testPredict

plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

