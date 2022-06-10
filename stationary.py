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

from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(df)
print("pvalue=", pvalue, " if above 0.5, data is not stationary")

df['year'] = [d.year for d in df.index]
df['month'] = [d.strftime('%b') for d in df.index]
years= df['year'].unique()

sns.boxplot(x='year', y= 'Passengers', data = df)
sns.boxplot(x='month', y= 'Passengers', data = df)


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


from statsmodels.tsa.stattools import acf

acf_144 = acf(df.Passengers, nlags= 144)
plt.plot(acf_144)


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df.Passengers)