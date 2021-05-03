# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:09:07 2021

@author: Francesco Conforte
"""


import pymongo as pm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore',category=FutureWarning)
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA
import seaborn as sb


Calgary = pd.read_csv('calgary.csv',sep=',',parse_dates=[0],index_col=0)
Milano = pd.read_csv('milano.csv',sep=',',parse_dates=[0],index_col=0)
Amsterdam = pd.read_csv('amsterdam.csv',sep=',',parse_dates=[0],index_col=0)



#%% [OPTIONAL]
#Milano
X_mi = Milano.values.astype(float)
tr_size_mi = 312 #best size train (312h=13 days)
ts_size_mi = 72 #size test (72h = 3 days)
h_values = range(1,25)

predictions=np.zeros((len(h_values),ts_size_mi))
predictions.fill(np.nan)
MAE=np.zeros((len(h_values)))
MSE=np.zeros((len(h_values)))
MAPE=np.zeros((len(h_values)))
R2=np.zeros((len(h_values)))
train_mi, test_mi = X_mi[0:tr_size_mi], X_mi[tr_size_mi:(tr_size_mi+ts_size_mi)]


for h in h_values:
    history = [x for x in train_mi]
    for t in range(0, ts_size_mi):#for each hour I do arima model
        model = ARIMA(history, order=(4,0,1))
        model_fit = model.fit(disp=0, method='css')
        output,_,_ = model_fit.forecast(steps=h) #get all the forecast
         
        yhat = output[-1] #first forecast
        if t+h>72:
            break
        #     yhat=output[:(72-t)]
        predictions[h_values.index(h)][t+h-1]=yhat
         
        obs = test_mi[t]
        history.extend(obs)
        
    MAE[h_values.index(h)]=mean_absolute_error(test_mi[h-1:], predictions[h_values.index(h)][h-1:])
    MSE[h_values.index(h)]=mean_squared_error(test_mi[h-1:], predictions[h_values.index(h)][h-1:])
    MAPE[h_values.index(h)]=mean_absolute_percentage_error(
        test_mi[h-1:], predictions[h_values.index(h)][h-1:])
    R2[h_values.index(h)]=r2_score(test_mi[h-1:], predictions[h_values.index(h)][h-1:])


fig = plt.figure(figsize=(14, 8))
x=1
for i in predictions:
    plt.plot(i,label=('h='+str(x)))
    x+=1
plt.plot(test_mi,color='black',label='original')
plt.xlabel('Test Hours')
plt.ylabel('Rentals')
plt.title('Milano: predictions for different forecast horizons')
plt.grid()
fig.legend()


fig, ax1 = plt.subplots()
ax1.set_xlabel('h')
ax1.set_ylabel('MAPE')
ax1.plot(h_values, MAPE, color='red', label='MAPE')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('MSE')  # we already handled the x-label with ax1
ax2.plot(h_values, MSE, color='blue', label='MSE')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
fig.legend()
plt.title('MAPE and MSE with different time horizons')
plt.show()


train = Milano.loc['2017-01-30 00:00:00':'2017-02-02 23:59:59'].astype(float)
model = ARIMA(train, order=(4,0,1))
model_fit = model.fit(disp=0)
#print(model_fit.summary())


fig, ax = plt.subplots()
ax = Milano.loc['2017-01-30 00:00:00':].plot(ax=ax)

#plot data for future day
fig = model_fit.plot_predict('2017-01-30 00:00:00', '2017-02-05 23:59:59', dynamic=False,
                             ax=ax, plot_insample=False)




#Calgary
X_ca = Calgary.values.astype(float)
tr_size_ca = 144 #best size train (312h=13 days)
ts_size_ca = 72 #size test (72h = 3 days)
h_values = range(1,25)

predictions=np.zeros((len(h_values),ts_size_ca))
predictions.fill(np.nan)
MAE=np.zeros((len(h_values)))
MSE=np.zeros((len(h_values)))
MAPE=np.zeros((len(h_values)))
R2=np.zeros((len(h_values)))
train_ca, test_ca = X_ca[0:tr_size_ca], X_ca[tr_size_ca:(tr_size_ca+ts_size_ca)]


for h in h_values:
    history = [x for x in train_ca]
    for t in range(0, ts_size_ca):#for each hour I do arima model
        model = ARIMA(history, order=(5,0,2))
        model_fit = model.fit(disp=0, method='css')
        output,_,_ = model_fit.forecast(steps=h) #get all the forecast
         
        yhat = output[-1] #first forecast
        if t+h>72:
            break
        #     yhat=output[:(72-t)]
        predictions[h_values.index(h)][t+h-1]=yhat
         
        obs = test_ca[t]
        history.extend(obs)
        
    MAE[h_values.index(h)]=mean_absolute_error(test_ca[h-1:], predictions[h_values.index(h)][h-1:])
    MSE[h_values.index(h)]=mean_squared_error(test_ca[h-1:], predictions[h_values.index(h)][h-1:])
    MAPE[h_values.index(h)]=mean_absolute_percentage_error(
        test_ca[h-1:], predictions[h_values.index(h)][h-1:])
    R2[h_values.index(h)]=r2_score(test_ca[h-1:], predictions[h_values.index(h)][h-1:])


fig = plt.figure(figsize=(14, 8))
x=1
for i in predictions:
    plt.plot(i,label=('h='+str(x)))
    x+=1
plt.plot(test_ca,color='black',label='original')
plt.xlabel('Test Hours')
plt.ylabel('Rentals')
plt.title('Calgary: predictions for different forecast horizons')
plt.grid()
fig.legend()


fig, ax1 = plt.subplots()
ax1.set_xlabel('h')
ax1.set_ylabel('MAPE')
ax1.plot(h_values, MAPE, color='red', label='MAPE')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('MSE')  # we already handled the x-label with ax1
ax2.plot(h_values, MSE, color='blue', label='MSE')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
fig.legend()
plt.title('MAPE and MSE with different time horizons')
plt.show()


train = Calgary.loc['2017-01-30 00:00:00':'2017-02-02 23:59:59'].astype(float)
model = ARIMA(train, order=(5,0,2))
model_fit = model.fit(disp=0)
#print(model_fit.summary())


fig, ax = plt.subplots()
ax = Calgary.loc['2017-01-30 00:00:00':].plot(ax=ax)

#plot data for future day
fig = model_fit.plot_predict('2017-01-30 00:00:00', '2017-02-05 23:59:59', dynamic=False,
                             ax=ax, plot_insample=False)



#Amsterdam
X_am = Amsterdam.values.astype(float)
tr_size_am = 480 #best size train (312h=13 days)
ts_size_am = 72 #size test (72h = 3 days)
h_values = range(1,25)

predictions=np.zeros((len(h_values),ts_size_am))
predictions.fill(np.nan)
MAE=np.zeros((len(h_values)))
MSE=np.zeros((len(h_values)))
MAPE=np.zeros((len(h_values)))
R2=np.zeros((len(h_values)))
train_am, test_am = X_am[0:tr_size_am], X_am[tr_size_am:(tr_size_am+ts_size_am)]


for h in h_values:
    history = [x for x in train_am]
    for t in range(0, ts_size_am):#for each hour I do arima model
        model = ARIMA(history, order=(5,0,2))
        model_fit = model.fit(disp=0, method='css')
        output,_,_ = model_fit.forecast(steps=h) #get all the forecast
         
        yhat = output[-1] #first forecast
        if t+h>72:
            break
        #     yhat=output[:(72-t)]
        predictions[h_values.index(h)][t+h-1]=yhat
         
        obs = test_am[t]
        history.extend(obs)
        
    MAE[h_values.index(h)]=mean_absolute_error(test_am[h-1:], predictions[h_values.index(h)][h-1:])
    MSE[h_values.index(h)]=mean_squared_error(test_am[h-1:], predictions[h_values.index(h)][h-1:])
    MAPE[h_values.index(h)]=mean_absolute_percentage_error(
        test_am[h-1:], predictions[h_values.index(h)][h-1:])
    R2[h_values.index(h)]=r2_score(test_am[h-1:], predictions[h_values.index(h)][h-1:])


fig = plt.figure(figsize=(14, 8))
x=1
for i in predictions:
    plt.plot(i,label=('h='+str(x)))
    x+=1
plt.plot(test_am,color='black',label='original')
plt.xlabel('Test Hours')
plt.ylabel('Rentals')
plt.title('Amsterdam: predictions for different forecast horizons')
plt.grid()
fig.legend()


fig, ax1 = plt.subplots()
ax1.set_xlabel('h')
ax1.set_ylabel('MAPE')
ax1.plot(h_values, MAPE, color='red', label='MAPE')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('MSE')  # we already handled the x-label with ax1
ax2.plot(h_values, MSE, color='blue', label='MSE')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
fig.legend()
plt.title('MAPE and MSE with different time horizons')
plt.show()


train = Amsterdam.loc['2017-01-30 00:00:00':'2017-02-02 23:59:59'].astype(float)
model = ARIMA(train, order=(5,0,2))
model_fit = model.fit(disp=0)
#print(model_fit.summary())


fig, ax = plt.subplots()
ax = Amsterdam.loc['2017-01-30 00:00:00':].plot(ax=ax)

#plot data for future day
fig = model_fit.plot_predict('2017-01-30 00:00:00', '2017-02-05 23:59:59', dynamic=False,
                             ax=ax, plot_insample=False)