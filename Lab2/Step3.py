# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:58:26 2020

@author: Francesco Conforte
"""
import pymongo as pm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA
import seaborn as sb

warnings.filterwarnings('ignore')


#%% Connection to the server
client = pm.MongoClient('bigdatadb.polito.it',
                        ssl=True,
                        authSource = 'carsharing',
                        tlsAllowInvalidCertificates=True)
db = client['carsharing'] #Choose the DB to use
db.authenticate('ictts', 'Ictts16!')#, mechanism='MONGODB-CR') #authentication


#%%Extraction of times series from MongoDB server
start = datetime(2017,1,7,0,0,0) 
end = datetime(2017,2,5,23,59,59)

cars_per_hour_filtered_list=[]
cities = ["Milano", "Calgary", "Amsterdam"]
for c in cities:
    cars_per_hour_filtered = db.get_collection("PermanentBookings").aggregate(
          [
              { "$match" : {"$and": [ { "city": c },
                                      { "init_date": { "$gte": start } },
                                      { "final_date": { "$lte": end } },
                                  ]
                          } 
              },
              { "$project": {
                    "_id": 1,
                    "city": 1,
                    "moved": { "$ne": [
                          {"$arrayElemAt": [ "$origin_destination.coordinates", 0]},
                          {"$arrayElemAt": [ "$origin_destination.coordinates", 1]} 
                      ]
                    },
                    "duration": { "$divide": [ { "$subtract": ["$final_time", 
                                                               "$init_time"] }, 60 ] },
                    "date_parts": { "$dateToParts": { "date": "$init_date" } },
                }
              },
              { "$match" : { "$and": [ { "duration": { "$gte": 3 } },
                                        { "duration": { "$lte": 180 } },
                                        { "moved": True }
                                      ] 
                            }
              },
              { "$group": {
                    "_id": {
                      "day": "$date_parts.day",
                      "month":"$date_parts.month",
                      "hour": "$date_parts.hour"
                    },
                    "tot_rentals": {"$sum": 1}
              }
              },
              { "$sort": {"_id": 1} }
          ]
        )
    cars_per_hour_filtered_list.append(list(cars_per_hour_filtered))
    
cars_per_hour_Milano = cars_per_hour_filtered_list[0]
cars_per_hour_Calgary = cars_per_hour_filtered_list[1]
cars_per_hour_Amsterdam = cars_per_hour_filtered_list[2]


#%%check missing data for Calgary
days=[]  
rentals_Calgary=[]   
for i in range(len(cars_per_hour_Calgary)):
    days.append(datetime(2017,cars_per_hour_Calgary[i]['_id']['month'],
                          cars_per_hour_Calgary[i]['_id']['day'],
                          cars_per_hour_Calgary[i]['_id']['hour']))
    rentals_Calgary.append(cars_per_hour_Calgary[i]['tot_rentals'])


Calgary=pd.DataFrame({'days':days,'rentals':rentals_Calgary},index=days) 
missing_Calgary=pd.date_range(start = '2017-01-07', end = '2017-02-05', 
                              freq='H' ).difference(Calgary.index) #find missing hours
Calgary=Calgary.sort_index()
#add missing data for Calgary
Nr=missing_Calgary.shape[0]
if Nr>0:
    missing_data=pd.DataFrame({'days':missing_Calgary,'rentals':np.full(Nr,-1)},
                              index=missing_Calgary.values)#add 0 as temp value
    Calgary=Calgary.append(missing_data)#add the two missing values to main Dataset
    Calgary=Calgary.sort_index()#sort rows to put new rows to right place
    index=np.argwhere((Calgary['rentals'].values==-1)) #find index where rentals are 0
    Calgary['rentals'].values[index]=(Calgary['rentals'].values[index-1]+
                                Calgary['rentals'].values[index+1])//2 #use mean values


#%%check missing data for Milano
days=[] 
rentals_Milano=[]    
for i in range(len(cars_per_hour_Milano)):
    days.append(datetime(2017,cars_per_hour_Milano[i]['_id']['month'],
                          cars_per_hour_Milano[i]['_id']['day'],
                          cars_per_hour_Milano[i]['_id']['hour']))
    rentals_Milano.append(cars_per_hour_Milano[i]['tot_rentals'])


Milano=pd.DataFrame({'days':days,'rentals':rentals_Milano},index=days)
missing_Milano=pd.date_range(start = '2017-01-07', end = '2017-02-05', 
                              freq='H' ).difference(Milano.index)
Milano=Milano.sort_index()
#add missing data for Milano
Nr=missing_Milano.shape[0]
if Nr>0:
    missing_data=pd.DataFrame({'days':missing_Milano,'rentals':np.full(Nr,-1)},
                              index=missing_Milano.values)#add 0 as temp value
    Milano=Milano.append(missing_data)#add the two missing values to main Dataset
    Milano=Milano.sort_index()#sort rows to put new rows to right place
    index=np.argwhere((Milano['rentals'].values==-1)) #find index where rentals are 0
    Milano['rentals'].values[index]=(Milano['rentals'].values[index-1]+
                                Milano['rentals'].values[index+1])//2 #use mean values


#%%check missing data for Amsterdam
days=[] 
rentals_Amsterdam=[]    
for i in range(len(cars_per_hour_Amsterdam)):
    days.append(datetime(2017,cars_per_hour_Amsterdam[i]['_id']['month'],
                          cars_per_hour_Amsterdam[i]['_id']['day'],
                          cars_per_hour_Amsterdam[i]['_id']['hour']))
    rentals_Amsterdam.append(cars_per_hour_Amsterdam[i]['tot_rentals'])

Amsterdam=pd.DataFrame({'days':days,'rentals':rentals_Amsterdam},index=days)
missing_Amsterdam=pd.date_range(start = '2017-01-07', end = '2017-02-05', 
                                freq='H' ).difference(Amsterdam.index)
Amsterdam=Amsterdam.sort_index()

#add missing data for Amsterdam
Nr=missing_Amsterdam.shape[0]
if Nr>0:
    missing_data=pd.DataFrame({'days':missing_Amsterdam,'rentals':np.full(Nr,-1)},
                              index=missing_Amsterdam.values)#add 0 as temp value
    Amsterdam=Amsterdam.append(missing_data)#add the two missing values to main Dataset
    Amsterdam=Amsterdam.sort_index()#sort rows to put new rows to right place
    index=np.argwhere((Amsterdam['rentals'].values==-1)) #find index where rentals are 0
    Amsterdam['rentals'].values[index]=(Amsterdam['rentals'].values[index-1]+
                                Amsterdam['rentals'].values[index+1])//2 #use mean values

Calgary.to_csv('calgary.csv', index=False)
Milano.to_csv('milano.csv', index=False)
Amsterdam.to_csv('amsterdam.csv', index=False)

Calgary = pd.read_csv('calgary.csv',sep=',',parse_dates=[0],index_col=0)
Milano = pd.read_csv('milano.csv',sep=',',parse_dates=[0],index_col=0)
Amsterdam = pd.read_csv('amsterdam.csv',sep=',',parse_dates=[0],index_col=0)

#%% CHECK OF STATIONARITY

fig, ax = plt.subplots()
ax.plot(Calgary.index.values,Calgary["rentals"].values,label='real rentals')
rolling_mean_calgary=Calgary['rentals'].rolling(168).mean()# 168 is 24*1week
rolling_std_calgary=Calgary['rentals'].rolling(168).std()# 168 is 24*1week
ax.plot(rolling_mean_calgary,label='rolling mean')
ax.plot(rolling_std_calgary,label='rolling std')
ax.legend(ncol=1,fontsize='small')
fig.autofmt_xdate()
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.title('Rolling statistics Calgary (1 week sliding window)')
plt.show()


fig, ax = plt.subplots()
ax.plot(Milano.index.values,Milano["rentals"].values,label='real rentals')
rolling_mean_milano=Milano['rentals'].rolling(168).mean()
rolling_std_milano=Milano['rentals'].rolling(168).std()# 168 is 24*1week
ax.plot(rolling_mean_milano,label='rolling mean')
ax.plot(rolling_std_milano,label='rolling std')
ax.legend(ncol=1,fontsize='small')
fig.autofmt_xdate()
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.title('Rolling statistics Milano (1 week sliding window)')
plt.show()

fig, ax = plt.subplots()
ax.plot(Amsterdam.index.values,Amsterdam["rentals"].values,label='real rentals')
rolling_mean_amsterdam=Amsterdam['rentals'].rolling(168).mean()
rolling_std_amsterdam=Amsterdam['rentals'].rolling(168).std()# 168 is 24*1week
ax.plot(rolling_mean_amsterdam,label='rolling mean')
ax.plot(rolling_std_amsterdam,label='rolling std')
ax.legend(ncol=1,fontsize='small')
fig.autofmt_xdate()
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.title('Rolling statistics Amsterdam (1 week sliding window)')
plt.show()


#%% ACF AND PACF
#Plot ACF and PACF for Amsterdam
sm.graphics.tsa.plot_acf(Amsterdam.rentals.values, lags=40,
                         title='Autocorrelation for Amsterdam')
sm.graphics.tsa.plot_pacf(Amsterdam.rentals.values, lags=40,
                          title='Partial Autocorrelation for Amsterdam')
plt.show()

#Plot ACF and PACF for Calgary
sm.graphics.tsa.plot_acf(Calgary.rentals.values, lags=40,
                         title='Autocorrelation for Calgary')
sm.graphics.tsa.plot_pacf(Calgary.rentals.values, lags=40,
                          title='Partial Autocorrelation for Calgary')
plt.show()

#Plot ACF and PACF for Milano
sm.graphics.tsa.plot_acf(Milano.rentals.values, lags=40,
                         title='Autocorrelation for Milano')
sm.graphics.tsa.plot_pacf(Milano.rentals.values, lags=40,
                          title='Partial Autocorrelation for Milano')
plt.show()


#%% FIT THE MODEL without division in test and training
#Amsterdam
model = ARIMA(Amsterdam.astype(float), order=(2,0,4))
model_fit = model.fit(disp=0,method='css')
print(model_fit.summary())

fig, ax = plt.subplots()
ax.plot(Amsterdam.index.values,Amsterdam.rentals.values,label='real rentals')
ax.plot(model_fit.fittedvalues, color='red', label='predicted values')
ax.legend(ncol=1,fontsize='small')
fig.autofmt_xdate()
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.title('Real reltals vs Predicted Values with ARIMA(2,0,4) for Amsterdam')
plt.show()


#Calgary
model = ARIMA(Calgary.astype(float), order=(2,0,5))
model_fit = model.fit(disp=0,method='css')
print(model_fit.summary())

fig, ax = plt.subplots()
ax.plot(Calgary.index.values,Calgary.rentals.values,label='real rentals')
ax.plot(model_fit.fittedvalues, color='red', label='predicted values')
ax.legend(ncol=1,fontsize='small')
fig.autofmt_xdate()
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.title('Real reltals vs Predicted Values with ARIMA(2,0,5) for Calgary')
plt.show()


#Milano
model = ARIMA(Milano.astype(float), order=(2,0,4))
model_fit = model.fit(disp=0,method='css')
print(model_fit.summary())

fig, ax = plt.subplots()
ax.plot(Milano.index.values,Milano.rentals.values,label='real rentals')
ax.plot(model_fit.fittedvalues, color='red', label='predicted values')
ax.legend(ncol=1,fontsize='small')
fig.autofmt_xdate()
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.title('Real reltals vs Predicted Values with ARIMA(2,0,5) for Milano')
plt.show()

#%%Fit the model with division in training and test
#Amsterdam

X_am = Amsterdam.values.astype(float)
tr_size_am = 504 #size train (504h=3 week)
ts_size_am = 72 #size test (72h = 3 days)

train_am, test_am = X_am[0:tr_size_am], X_am[tr_size_am:(tr_size_am+ts_size_am)]

history = [x for x in train_am]
predictions=[]
for t in range(0, ts_size_am):#for each hour I do arima model
    model = ARIMA(history, order=(3,0,4))
    model_fit = model.fit(disp=0,method='css')
    output = model_fit.forecast() #get all the forecast
     
    yhat = output[0] #first forecast
    predictions.append(yhat)
     
    obs = test_am[t]
    history.append(obs)
        
plt.figure()        
plt.plot(test_am, color='black',label='Original')
plt.plot(predictions, label = 'Prediction')
plt.xlabel('Test Hours')
plt.ylabel('Rentals')
plt.title('Amsterdam: ARIMA(3,0,4) with expanding window')
plt.legend() 
plt.show()      

print('Amsterdam: (3,0,4) model => MAE: %.3f -- MSE: %.3f -- MAPE: %.3f -- R2: %.3f' %(
    mean_absolute_error(test_am, predictions),
	mean_squared_error(test_am, predictions),
    mean_absolute_percentage_error(test_am, predictions),
	r2_score(test_am, predictions)))

#Calgary
X_ca = Calgary.values.astype(float)
tr_size_ca = 504 #size train (504h=3 week)
ts_size_ca = 72 #size test (72h = 3 days)

train_ca, test_ca = X_ca[0:tr_size_ca], X_ca[tr_size_ca:(tr_size_ca+ts_size_ca)]

history = [x for x in train_ca]
predictions=[]
for t in range(0, ts_size_ca):#for each hour I do arima model
    model = ARIMA(history, order=(2,0,5))
    model_fit = model.fit(disp=0,method='css')
    output = model_fit.forecast() #get all the forecast
     
    yhat = output[0] #first forecast
    predictions
    predictions.append(yhat)
     
    obs = test_ca[t]
    history.append(obs)
        
        
plt.figure()
plt.plot(test_ca, color='black',label='Original')
plt.plot(predictions, label = 'Prediction')
plt.xlabel('Test Hours')
plt.ylabel('Rentals')
plt.title('Calgary: ARIMA(2,0,5) with expanding window')
plt.legend() 
plt.show()

print('Calgary: (2,0,5) model => MAE: %.3f -- MSE: %.3f -- MAPE: %.3f -- R2: %.3f' %(
    mean_absolute_error(test_ca, predictions),
	mean_squared_error(test_ca, predictions),
    mean_absolute_percentage_error(test_ca, predictions),
	r2_score(test_ca, predictions)))

#Milano
X_mi = Milano.values.astype(float)
tr_size_mi = 504 #size train (504h=3 week)
ts_size_mi = 72 #size test (72h = 3 days)

train_mi, test_mi = X_mi[0:tr_size_mi], X_mi[tr_size_mi:(tr_size_mi+ts_size_mi)]

history = [x for x in train_mi]
predictions=[]
for t in range(0, ts_size_mi):#for each hour I do arima model
    model = ARIMA(history, order=(2,0,4))
    model_fit = model.fit(disp=0,method='css')
    output = model_fit.forecast() #get all the forecast
     
    yhat = output[0] #first forecast
    predictions.append(yhat)
     
    obs = test_mi[t]
    history.append(obs)
        
plt.figure()        
plt.plot(test_mi, color='black', label='Original')
plt.plot(predictions, label = 'Prediction')
plt.xlabel('Test Hours')
plt.ylabel('Rentals')
plt.title('Milano: ARIMA(2,0,4) with expanding window')
plt.legend()
plt.show()

print('Milano: (2,0,4) model => MAE: %.3f -- MSE: %.3f -- MAPE: %.3f -- R2: %.3f' %(
    mean_absolute_error(test_mi, predictions),
	mean_squared_error(test_mi, predictions),
    mean_absolute_percentage_error(test_mi, predictions),
	r2_score(test_mi, predictions)))


#%%Fit the model with a grid search over p and q with train and test fixed
##Amsterdam
p_values = range(0,8) #y axis
q_values = range(0,6) #x axis

X_am = Amsterdam.values.astype(float)
tr_size_am = 504 #size train (504h=3 week)
ts_size_am = 72 #size test (72h = 3 days)

train_am, test_am = X_am[0:tr_size_am], X_am[tr_size_am:(tr_size_am+ts_size_am)]


predictions=np.zeros((len(p_values),len(q_values),ts_size_am))
MAE=np.zeros((len(p_values),len(q_values)))
MSE=np.zeros((len(p_values),len(q_values)))
MAPE=np.zeros((len(p_values),len(q_values)))
R2=np.zeros((len(p_values),len(q_values)))

warnings.filterwarnings('ignore')
for p in p_values:
    for q in q_values:
        if (p==0 and q==0):
            MAE[p_values.index(p)][q_values.index(q)]=np.nan
            MSE[p_values.index(p)][q_values.index(q)]=np.nan
            MAPE[p_values.index(p)][q_values.index(q)]=np.nan
            R2[p_values.index(p)][q_values.index(q)]=np.nan
            continue
        print('Testing ARIMA order (%i,0,%i)' %(p,q))
        history = [x for x in train_am]
        flag=0
        for t in range(0, ts_size_am):#for each hour I do arima model
            model = ARIMA(history, order=(p,0,q))
            try:
              model_fit = model.fit(disp=0, method='css')
            except ValueError:
              flag=1
              MAE[p_values.index(p)][q_values.index(q)]=np.nan
              MSE[p_values.index(p)][q_values.index(q)]=np.nan
              MAPE[p_values.index(p)][q_values.index(q)]=np.nan
              R2[p_values.index(p)][q_values.index(q)]=np.nan
              break
            output = model_fit.forecast() #get all the forecast
             
            yhat = output[0] #first forecast
            predictions[p_values.index(p)][q_values.index(q)][t]=yhat
             
            obs = test_am[t]
            history.append(obs)
        if flag==0:
            MAE[p_values.index(p)][q_values.index(q)]=mean_absolute_error(
                test_am, predictions[p_values.index(p)][q_values.index(q)])
            MSE[p_values.index(p)][q_values.index(q)]=mean_squared_error(
                test_am, predictions[p_values.index(p)][q_values.index(q)])
            MAPE[p_values.index(p)][q_values.index(q)]=mean_absolute_percentage_error(
                test_am, predictions[p_values.index(p)][q_values.index(q)])
            R2[p_values.index(p)][q_values.index(q)]=r2_score(
                test_am, predictions[p_values.index(p)][q_values.index(q)])

plt.figure()
heat_map = sb.heatmap(MAPE,xticklabels=q_values,yticklabels=p_values,annot=True,
                      cmap='gist_stern')
heat_map.set_title('MAPE: Expanding window for different value of p and q Amsterdam')
heat_map.set_xlabel('q')
heat_map.set_ylabel('p')
plt.show()

ind=np.argwhere(MAPE == np.nanmin(MAPE))[0]
print('-----Amsterdam-----')
print('p best is: ' + str(p_values[ind[0]]))
print('q best is: ' + str(q_values[ind[1]]))

##Calgary
p_values = range(0,8) #y axis
q_values = range(0,6) #x axis


X_ca = Calgary.values.astype(float)
tr_size_ca = 504 #size train (504h=3 week)
ts_size_ca = 72 #size test (72h = 3 days)

train_ca, test_ca = X_ca[0:tr_size_ca], X_ca[tr_size_ca:(tr_size_ca+ts_size_ca)]


predictions=np.zeros((len(p_values),len(q_values),ts_size_ca))
MAE=np.zeros((len(p_values),len(q_values)))
MSE=np.zeros((len(p_values),len(q_values)))
MAPE=np.zeros((len(p_values),len(q_values)))
R2=np.zeros((len(p_values),len(q_values)))

warnings.filterwarnings('ignore')
for p in p_values:
    for q in q_values:
        if (p==0 and q==0):
            MAE[p_values.index(p)][q_values.index(q)]=np.nan
            MSE[p_values.index(p)][q_values.index(q)]=np.nan
            MAPE[p_values.index(p)][q_values.index(q)]=np.nan
            R2[p_values.index(p)][q_values.index(q)]=np.nan
            continue
        print('Testing ARIMA order (%i,0,%i)' %(p,q))
        history = [x for x in train_ca]
        flag=0
        for t in range(0, ts_size_ca):#for each hour I do arima model
            model = ARIMA(history, order=(p,0,q))
            try:
              model_fit = model.fit(disp=0,method='css')
            except ValueError:
              flag=1
              MAE[p_values.index(p)][q_values.index(q)]=np.nan
              MSE[p_values.index(p)][q_values.index(q)]=np.nan
              MAPE[p_values.index(p)][q_values.index(q)]=np.nan
              R2[p_values.index(p)][q_values.index(q)]=np.nan
              break
            output = model_fit.forecast() #get all the forecast
             
            yhat = output[0] #first forecast
            predictions[p_values.index(p)][q_values.index(q)][t]=yhat
             
            obs = test_ca[t]
            history.append(obs)
        if flag==0:
            MAE[p_values.index(p)][q_values.index(q)]=mean_absolute_error(
                test_ca, predictions[p_values.index(p)][q_values.index(q)])
            MSE[p_values.index(p)][q_values.index(q)]=mean_squared_error(
                test_ca, predictions[p_values.index(p)][q_values.index(q)])
            MAPE[p_values.index(p)][q_values.index(q)]=mean_absolute_percentage_error(
                test_ca, predictions[p_values.index(p)][q_values.index(q)])
            R2[p_values.index(p)][q_values.index(q)]=r2_score(
                test_ca, predictions[p_values.index(p)][q_values.index(q)])

plt.figure()
heat_map = sb.heatmap(MAPE,xticklabels=q_values,yticklabels=p_values,annot=True,
                      cmap='gist_stern')
heat_map.set_title('MAPE: Expanding window for different value of p and q Calgary')
heat_map.set_xlabel('q')
heat_map.set_ylabel('p')
plt.show()

ind=np.argwhere(MAPE == np.nanmin(MAPE))[0]
print('-----Calgary-----')
print('p best is: ' + str(p_values[ind[0]]))
print('q best is: ' + str(q_values[ind[1]]))


##Milano
p_values = range(0,8) #y axis
q_values = range(0,6) #x axis


X_mi = Milano.values.astype(float)
tr_size_mi = 504 #size train (504h=3 week)
ts_size_mi = 72 #size test (72h = 3 days)

train_mi, test_mi = X_mi[0:tr_size_mi], X_mi[tr_size_mi:(tr_size_mi+ts_size_mi)]


predictions=np.zeros((len(p_values),len(q_values),ts_size_mi))
MAE=np.zeros((len(p_values),len(q_values)))
MSE=np.zeros((len(p_values),len(q_values)))
MAPE=np.zeros((len(p_values),len(q_values)))
R2=np.zeros((len(p_values),len(q_values)))

warnings.filterwarnings('ignore')
for p in p_values:
    for q in q_values:
        if (p==0 and q==0):
            MAE[p_values.index(p)][q_values.index(q)]=np.nan
            MSE[p_values.index(p)][q_values.index(q)]=np.nan
            MAPE[p_values.index(p)][q_values.index(q)]=np.nan
            R2[p_values.index(p)][q_values.index(q)]=np.nan
            continue
        print('Testing ARIMA order (%i,0,%i)' %(p,q))
        history = [x for x in train_mi]
        flag=0
        for t in range(0, ts_size_mi):#for each hour I do arima model
            model = ARIMA(history, order=(p,0,q))
            try:
              model_fit = model.fit(disp=0,method='css')
            except ValueError:
              flag=1
              MAE[p_values.index(p)][q_values.index(q)]=np.nan
              MSE[p_values.index(p)][q_values.index(q)]=np.nan
              MAPE[p_values.index(p)][q_values.index(q)]=np.nan
              R2[p_values.index(p)][q_values.index(q)]=np.nan
              break
            output = model_fit.forecast() #get all the forecast
             
            yhat = output[0] #first forecast
            predictions[p_values.index(p)][q_values.index(q)][t]=yhat
             
            obs = test_mi[t]
            history.append(obs)
        if flag==0:
            MAE[p_values.index(p)][q_values.index(q)]=mean_absolute_error(
                test_mi, predictions[p_values.index(p)][q_values.index(q)])
            MSE[p_values.index(p)][q_values.index(q)]=mean_squared_error(
                test_mi, predictions[p_values.index(p)][q_values.index(q)])
            MAPE[p_values.index(p)][q_values.index(q)]=mean_absolute_percentage_error(
                test_mi, predictions[p_values.index(p)][q_values.index(q)])
            R2[p_values.index(p)][q_values.index(q)]=r2_score(
                test_mi, predictions[p_values.index(p)][q_values.index(q)])

plt.figure()
heat_map = sb.heatmap(MAPE,xticklabels=q_values,yticklabels=p_values,annot=True,
                      cmap='gist_stern')
heat_map.set_title('MAPE: Expanding window for different value of p and q Milano')
heat_map.set_xlabel('q')
heat_map.set_ylabel('p')
plt.show()

ind=np.argwhere(MAPE == np.nanmin(MAPE))[0]
print('-----Milano-----')
print('p best is: ' + str(p_values[ind[0]]))
print('q best is: ' + str(q_values[ind[1]]))

#%%Fit the model for different sizes of training. Expanding + Sliding window
##Amsterdam
X_am = Amsterdam.values.astype(float)
methods = [0,1] #0 expanding; 1 sliding
tr_size_am = 504 #size train (504h=3 week)
ts_size_am = 72 #size test (72h = 3 days)

N_values = list(np.linspace(72,tr_size_am,num=19, dtype=int)) 
predictions=np.zeros((len(N_values),len(methods),ts_size_am))
MAE=np.zeros((len(N_values),len(methods)))
MSE=np.zeros((len(N_values),len(methods)))
MAPE=np.zeros((len(N_values),len(methods)))
R2=np.zeros((len(N_values),len(methods)))

print('=====Amsterdam=====')
for m in methods:
    for j in N_values:
        print('Testing method %s and training size %i' %(m,j))
        train_am, test_am = X_am[0:j], X_am[j:(j+ts_size_am)]
        history = [x for x in train_am]
        for t in range(0, ts_size_am):#for each hour I do arima model
            model = ARIMA(history, order=(4,0,2))
            model_fit = model.fit(disp=0, method='css')
            output = model_fit.forecast() #get all the forecast
             
            yhat = output[0] #first forecast
            predictions[N_values.index(j)][methods.index(m)][t]=yhat
             
            obs = test_am[t]
            history.append(obs)
            history=history[m:]
        MAE[N_values.index(j)][methods.index(m)]=mean_absolute_error(
            test_am, predictions[N_values.index(j)][methods.index(m)])
        MSE[N_values.index(j)][methods.index(m)]=mean_squared_error(
            test_am, predictions[N_values.index(j)][methods.index(m)])
        MAPE[N_values.index(j)][methods.index(m)]=mean_absolute_percentage_error(
            test_am, predictions[N_values.index(j)][methods.index(m)])
        R2[N_values.index(j)][methods.index(m)]=r2_score(
            test_am, predictions[N_values.index(j)][methods.index(m)])

plt.figure()
heat_map = sb.heatmap(MAPE,xticklabels=methods,yticklabels=N_values,annot=True)
heat_map.set_xticklabels(['Expanding','Sliding'])
heat_map.set_ylabel('Number of Training Values [Hours]')
plt.suptitle('MAPE for different sizes of training and with \n expanding and \
             sliding methods for Amsterdam')
plt.show


plt.figure()
plt.plot(N_values, MAPE[:,0],label='Expanding')
plt.plot(N_values, MAPE[:,1],label='Sliding')
plt.xlabel('Number of Training Values [Hours]')
plt.ylabel('MAPE')
plt.title('MAPE vs Learning strategy and Training size Amsterdam')
plt.legend()
plt.grid()


##Calgary
X_ca = Calgary.values.astype(float)
methods = [0,1] #0 expanding; 1 sliding
tr_size_ca = 504 #size train (504h=3 week)
ts_size_ca = 72 #size test (72h = 3 days)

N_values = list(np.linspace(72,tr_size_ca,num=19, dtype=int)) 
predictions=np.zeros((len(N_values),len(methods),ts_size_ca))
MAE=np.zeros((len(N_values),len(methods)))
MSE=np.zeros((len(N_values),len(methods)))
MAPE=np.zeros((len(N_values),len(methods)))
R2=np.zeros((len(N_values),len(methods)))

print('=====Calgary=====')
warnings.filterwarnings('ignore')
for m in methods:
    for j in N_values:
        print('Testing method %s and training size %i' %(m,j))
        train_ca, test_ca = X_ca[0:j], X_ca[j:(j+ts_size_ca)]
        history = [x for x in train_ca]
        for t in range(0, ts_size_ca):#for each hour I do arima model
            model = ARIMA(history, order=(5,0,2))
            model_fit = model.fit(disp=0, method='css')
            output = model_fit.forecast() #get all the forecast
             
            yhat = output[0] #first forecast
            predictions[N_values.index(j)][methods.index(m)][t]=yhat
             
            obs = test_ca[t]
            history.append(obs)
            history=history[m:]
        MAE[N_values.index(j)][methods.index(m)]=mean_absolute_error(
            test_ca, predictions[N_values.index(j)][methods.index(m)])
        MSE[N_values.index(j)][methods.index(m)]=mean_squared_error(
            test_ca, predictions[N_values.index(j)][methods.index(m)])
        MAPE[N_values.index(j)][methods.index(m)]=mean_absolute_percentage_error(
            test_ca, predictions[N_values.index(j)][methods.index(m)])
        R2[N_values.index(j)][methods.index(m)]=r2_score(
            test_ca, predictions[N_values.index(j)][methods.index(m)])

plt.figure()
heat_map = sb.heatmap(MAPE,xticklabels=methods,yticklabels=N_values,annot=True)
heat_map.set_xticklabels(['Expanding','Sliding'])
heat_map.set_ylabel('Number of Training Values [Hours]')
plt.suptitle('MAPE for different sizes of training and with \n expanding and \
             sliding methods for Calgary')
plt.show()

plt.figure()
plt.plot(N_values, MAPE[:,0],label='Expanding')
plt.plot(N_values, MAPE[:,1],label='Sliding')
plt.xlabel('Number of Training Values [Hours]')
plt.ylabel('MAPE')
plt.title('MAPE vs Learning strategy and Training size Calgary')
plt.legend()
plt.grid()


##Milano
X_mi = Milano.values.astype(float)
methods = [0,1] #0 expanding; 1 sliding
tr_size_mi = 504 #size train (504h=3 week)
ts_size_mi = 72 #size test (72h = 3 days)

N_values = list(np.linspace(72,tr_size_mi,num=19, dtype=int)) 
predictions=np.zeros((len(N_values),len(methods),ts_size_mi))
MAE=np.zeros((len(N_values),len(methods)))
MSE=np.zeros((len(N_values),len(methods)))
MAPE=np.zeros((len(N_values),len(methods)))
R2=np.zeros((len(N_values),len(methods)))

print('=====Milano=====')
for m in methods:
    for j in N_values:
        print('Testing method %s and training size %i' %(m,j))
        train_mi, test_mi = X_mi[0:j], X_mi[j:(j+ts_size_mi)]
        history = [x for x in train_mi]
        for t in range(0, ts_size_mi):#for each hour I do arima model
            model = ARIMA(history, order=(4,0,1))
            model_fit = model.fit(disp=0, method='css')
            output = model_fit.forecast() #get all the forecast
             
            yhat = output[0] #first forecast
            predictions[N_values.index(j)][methods.index(m)][t]=yhat
             
            obs = test_mi[t]
            history.append(obs)
            history=history[m:]
        MAE[N_values.index(j)][methods.index(m)]=mean_absolute_error(
            test_mi, predictions[N_values.index(j)][methods.index(m)])
        MSE[N_values.index(j)][methods.index(m)]=mean_squared_error(
            test_mi, predictions[N_values.index(j)][methods.index(m)])
        MAPE[N_values.index(j)][methods.index(m)]=mean_absolute_percentage_error(
            test_mi, predictions[N_values.index(j)][methods.index(m)])
        R2[N_values.index(j)][methods.index(m)]=r2_score(
            test_mi, predictions[N_values.index(j)][methods.index(m)])

plt.figure()
heat_map = sb.heatmap(MAPE,xticklabels=methods,yticklabels=N_values,annot=True)
heat_map.set_xticklabels(['Expanding','Sliding'])
heat_map.set_ylabel('Number of Training Values [Hours]')
plt.suptitle('MAPE for different sizes of training and with \n expanding and \
             sliding methods for Milano')
plt.show()

plt.figure()
plt.plot(N_values, MAPE[:,0],label='Expanding')
plt.plot(N_values, MAPE[:,1],label='Sliding')
plt.xlabel('Number of Training Values [Hours]')
plt.ylabel('MAPE')
plt.title('MAPE vs Learning strategy and Training size Milano')
plt.legend()
plt.grid()


#%% Final predictions

#Calgary
X_ca = Calgary.values.astype(float)
tr_size_ca = 144 #size train (144h = 6 days)
ts_size_ca = 72 #size test (72h = 3 days)

train_ca, test_ca = X_ca[0:tr_size_ca], X_ca[tr_size_ca:(tr_size_ca+ts_size_ca)]

history = [x for x in train_ca]
predictions=[]
for t in range(0, ts_size_ca):#for each hour I do arima model
    model = ARIMA(history, order=(5,0,2))
    model_fit = model.fit(disp=0,method='css')
    output = model_fit.forecast() #get all the forecast
     
    yhat = output[0] #first forecast
    predictions
    predictions.append(yhat)
     
    obs = test_ca[t]
    history.append(obs)
        
        
plt.figure()
plt.plot(test_ca, color='black',label='Original')
plt.plot(predictions, label = 'Prediction')
plt.xlabel('Test Hours')
plt.ylabel('Rentals')
plt.title('Calgary: ARIMA(5,0,2) best prediction')
plt.legend() 
plt.show()

#Amsterdam
X_am = Amsterdam.values.astype(float)
tr_size_am = 144 #size train (144h = 6 days)
ts_size_am = 72 #size test (72h = 3 days)

train_am, test_am = X_am[0:tr_size_am], X_am[tr_size_am:(tr_size_am+ts_size_am)]

history = [x for x in train_am]
predictions=[]
for t in range(0, ts_size_am):#for each hour I do arima model
    model = ARIMA(history, order=(4,0,2))
    model_fit = model.fit(disp=0,method='css')
    output = model_fit.forecast() #get all the forecast
     
    yhat = output[0] #first forecast
    predictions
    predictions.append(yhat)
     
    obs = test_am[t]
    history.append(obs)
        
        
plt.figure()
plt.plot(test_am, color='black',label='Original')
plt.plot(predictions, label = 'Prediction')
plt.xlabel('Test Hours')
plt.ylabel('Rentals')
plt.title('Amsterdam: ARIMA(4,0,2) best prediction')
plt.legend() 
plt.show()

#Milano
X_mi = Milano.values.astype(float)
tr_size_mi = 144 #size train (144h = 6 days)
ts_size_mi = 72 #size test (72h = 3 days)

train_mi, test_mi = X_mi[0:tr_size_mi], X_mi[tr_size_mi:(tr_size_mi+ts_size_mi)]

history = [x for x in train_mi]
predictions=[]
for t in range(0, ts_size_mi):#for each hour I do arima model
    model = ARIMA(history, order=(4,0,1))
    model_fit = model.fit(disp=0,method='css')
    output = model_fit.forecast() #get all the forecast
     
    yhat = output[0] #first forecast
    predictions
    predictions.append(yhat)
     
    obs = test_mi[t]
    history.append(obs)
        
        
plt.figure()
plt.plot(test_mi, color='black',label='Original')
plt.plot(predictions, label = 'Prediction')
plt.xlabel('Test Hours')
plt.ylabel('Rentals')
plt.title('Milano: ARIMA(4,0,1) best prediction')
plt.legend() 
plt.show()
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
plt.plot(test_mi,color='black')
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


