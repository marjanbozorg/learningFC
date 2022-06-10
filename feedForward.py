import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


#define model in a sequential way
from keras.models import Sequential
#connect layer to each other neural network part
from keras.layers import Dense
#ML package
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

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


#determine the input and the output as sequence
# considering 5 first ones and then predict the 6th
# do again consider from 2 till 6 as input and predict the 7th

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

print("Build deep model...")

#create and built dense model
model = Sequential()
model.add(Dense(64, input_dim=seq_size, activation='relu'))#12 too dense then overfitting
#only one model, it is optional
model.add(Dense(32, activation='relu'))#8
model.add(Dense(1))#y value and next value
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])#acc means accuracy
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

