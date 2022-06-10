import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('dark_background')

df = pd.read_csv('AirPassengers.csv')
print(df.dtypes)

df['Month'] = pd.to_datetime(df['Month'])
print(df.dtypes)

df.set_index('Month', inplace=True)

df = df.rename(columns={"#Passengers":"Passengers"})
print(df.head(5))

plt.plot(df["Passengers"])

##########section2

#ts is stationary or not, can use dicky-fuller test insted by adfuller
from pmdarima.arima import ADFTest
adf_test = ADFTest(alpha= 0.5)
adf_test.should_diff(df)

from statsmodels.tsa.seasonal import seasonal_decompose
decompsed = seasonal_decompose(df['Passengers'], model='additive')

trend = decompsed.trend
seasonal = decompsed.seasonal
residual = decompsed.resid


plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(df['Passengers'], label='Original', color='yellow')
plt.legend(loc= 'upper left')
plt.subplot(412)
plt.plot(trend, label='Trend', color='yellow')
plt.legend(loc= 'upper left')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal', color='yellow')
plt.legend(loc= 'upper left')
plt.subplot(414)
plt.plot(residual, label='Residual', color='yellow')
plt.legend(loc= 'upper left')
plt.show()


from pmdarima.arima import auto_arima


#p, d, q non-seasonal components
#P, D, Q seasonal compnents
arima_model = auto_arima(df['Passengers'], start_p= 1, d= 1, start_q= 1,
                         max_p= 5, max_q= 5, max_d= 5, m = 12,
                         start_P= 0, D= 1, start_Q= 0, max_P= 5, max_D= 5, max_Q= 5,
                         seasonal= True,
                         trace= True,
                         error_action= 'ignore',
                         suppress_warnings= True,
                         stepwise= True, n_fits= 50)

#should print the best parameters for p, d, q

print(arima_model.summary())
#should print the best model for me, here it will propose SARIMAX


#start FC
#separate training and testing data
# considering 2/3 of data as training
size = int(len(df)*0.66)
X_train, X_test = df[0:size], df[size:len(df)]

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(X_train['Passengers'],
                order= (0,1,1),
                seasonal_order= (2,1,1,12))

result = model.fit()
result.summary()

#Train prediction
start_index = 0
end_index = len(X_train)-1
train_prediction = result.predict(start_index, end_index)

#Prediction
start_index = len(X_train)
end_index = len(df)-1
prediction = result.predict(start_index, end_index).rename('Prediction passengers')

#plot prediction and actual values
prediction.plot(legend = True)
X_test['Passengers'].plot(legend = True)


import math
from sklearn.metrics import mean_squared_error
# calculate rmse
trainScore = math.sqrt(mean_squared_error(X_train, train_prediction))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(X_test, prediction))
print('Test Score: %.2f RMSE' % (testScore))

# for the next 3 year so, +3*12
forecast = result.predict(start = len(df),
                          end = (len(df)-1)+3 *12,
                          typ = 'levels'). rename('forecast') # for the next 3 year so, +3*12

plt.figure(figsize=(12,8))
plt.plot(X_train, label ='Training', color ='green')
plt.plot(X_test, label ='Test', color ='yellow')
plt.plot(forecast, label ='Forecast', color ='cyan')
plt.legend(loc='upper left')
plt.show()
