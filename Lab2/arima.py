import pandas as pandas
import matplotlib.pylab as plt
import numpy as np 

def dateparse (time_in_secs):
	return pd.datetime.fromtimestamp(float(time_in_secs))

df = pd.read_csv('torino.csv', sep='', parse_dates=[0],
	infer_datetime_format=True, date_parser=dateparse, index_col=0)
print(df.head())
print(df.dtypes)
# # #

df.plot(label='Time')
df.describe
# # #

#Select a stationary period of data - to avoid gaps and limits
df=df.loc[(df.index >= '2017-11-01 00:00:00') &
	(df.index <= '2017-12-01 00:00:00')]
df.plot()
# # #

#Fitting 
df2 = pd.DataFrame(columns = ['Time', 'rental'])
for i in range(1, len(df), 1):
	if(str(df.index[i]-df.index[i-1]) != '0 days 01:00:00'):
		#se la differenza è diversa da 1 ora
		steps = pd.date_range(df.index[i-1], df.index[i], freq='1H')
		for j in range(1, len(steps)-1):
			#cambiare i dati errati con i dati di un giorno prima
			#e l'aggiunta di una parte aleatoria
			new_value = df.iloc[i+j-24].rental + 10+np.random.random_sample()/100.0
			print('Fitting', i, j, new_value)
			df2.df2.append({'Time': steps[j], 'rental': new_value}, ignore_index=True)

df2 = df2.set_index('Time')
df = df.append(df2)
df = df.sort_index()
# # #

df.plot(label='Time')
# # #

#Check autocorrelation function 
pd.plotting.autocorrelation_plot(df)
#L'andamento è periodico su 24h quindi scegliamo un lag di 28
# # #

from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(df, nlags = 28) 
lag_pacf = pacf(df, nlags = 28, method = 'ols')

#Plot ACF
plt.subplot(211)
plt.plot(lag_acf)
plt.axis([0, 28, -0.5, 1])
plt.axhline(y=0, linestyle="--", color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df)), linestyle="--", color='gray')
plt.axhline(y=1.96/np.sqrt(len(df)), linestyle="--", color='gray')
plt.title('Autocorrelation function')

#Plot PACF
plt.subplot(212)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle="--", color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df)), linestyle="--", color='gray')
plt.axhline(y=1.96/np.sqrt(len(df)), linestyle="--", color='gray')
plt.title('Partial Autocorrelation function')
plt.tight_layout()
# # #

from statsmodels.tsa.arima_model import arima_model

#FIT MODEL
#fittare il modello con il parametri di p(in base al grafico di PACF) e q(in base a ACF)
model = ARIMA(df.astype(float), order=(4,0,1))
model_fit = model.fit(disp=0, method='css', maxiter=500)
print(model_fit.summary())

plt.plot(df)
plt.plot(model_fit.fittedvalues, color='red')
# # #

#PLOT RESIDUAL
residual = pd.DataFrame(model_fit.resid)
residual.plot()
residual.plot(kind='kde') #stima kernel della densità
print(residual.describe())
# # #

X = df.values.astype(float)
size = int(len(X) * 0.6) #amount of data for training
test_len = 72 #test for 3 days
lag_orders = (0,1,2,3,4) #values of p

predictions = np.zeros(len(lag_orders), test_len)

for p in lag_orders:
	print('Testing ARIMA order (%i,0,1' %p)

	train, test = X[0:size], X[size:(size+test_len)]

	history = [x for x in train]

	for t in range(0, test_len): #for each hour I do arima model
		model = ARIMA(history, order = (p,0,1))
		model_fit = model.fit(disp=0, method='css')
		output = model_fit.forecast() #get all the forecast

		yhat = output[0] #first forecast
		predictions[lag_orders.index(p)][t] = yhat

		obs = test[t]
		history.append(obs)
# # #

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

plt.plot(test, color='black', label='Orig')
for p in lag_orders:
	print('(%i,0,1) model => MAE: %.3f -- MSE: %.3f -- R2: %.3f' %(p, 
		mean_absolute_error(test, predictions[lag_orders.index(p)]),
		mean_squared_error(test, predictions[lag_orders.index(p)]),
		r2_score(test, predictions[lag_orders.index(p)])))
	plt.plot(predictions[lag_orders.index(p)], label = 'p=%i' %p)
plt.legend()
# # #

MA_orders = (0,1,2,3) #values of q
prediction = np.zeros(len(lag_orders), test_len)

for q in MA_orders:
	print('Testing ARIMA order (2,0,%i' %q)

	train, test = X[0:size], X[size:(size+test_len)]

	history = [x for x in train]

	for t in range(0, test_len): #for each hour I do arima model
		model = ARIMA(history, order = (2,0,q))
		model_fit = model.fit(disp=0, method='css')
		output = model_fit.forecast() #get all the forecast

		yhat = output[0] #first forecast
		predictions[lag_orders.index(q)][t] = yhat

		obs = test[t]
		history.append(obs)
# # #

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

plt.plot(test, color='black', label='Orig')
for q in MA_orders:
	print('(24,0,%i) model => MAE: %.3f -- MSE: %.3f -- R2: %.3f' %(q, 
		mean_absolute_error(test, predictions[lag_orders.index(q)]),
		mean_squared_error(test, predictions[lag_orders.index(q)]),
		r2_score(test, predictions[lag_orders.index(q)])))
	plt.plot(predictions[lag_orders.index(q)], label = 'q=%i' %q)
plt.legend()
# # #

#FINAL TESTING
model = ARIMA(df.astype(float), order=(2,0,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

plt.plot(df)
plt.plot(model_fit.fittedvalues, color='red')

print('MAE: %.3f -- MSE: %.3f -- R2: %.3f' % (
	mean_absolute_error(df, model_fit.fittedvalues),
	mean_squared_error(df, model_fit.fittedvalues),
	r2_score(df, model_fit.fittedvalues)))

fig, ax = plt.subplots()
ax = df.loc['2017-11-01 00:00:00':].plot(ax=ax)

#plot data for future day
fig = model_fit.plot_predict('2017-11-05 00:00:00', '2017-12-01', dynamic=True, ax=ax)
# # #

#FIT FINAL MODEL
p=24; d=0; q=1
model=ARIMA(df.astype(float), order=(p,d,q))
model_fit = model.fit(disp=0, method='css')
print(model_fit.summary())
# # #

#RESULTS
plt.plot(df)
plt.plot(model_fit.fittedvalues, color='red')
# # #

#FIT ANOTHER TIME
fig, ax = plt.subplots()
ax = df.loc['2017-11-01 00:00:00':].plot(ax=ax)

fig = model_fit.plot_predict('2017-11-05 00:00:00', '2017-11-15', dynamic=True, ax=ax) #stimiamo i prossimi giorni

residual = pd.DataFrame(model_fit.resid)
residual.plot()
residual.plot(kind='kde') #stima kernel della densità
print(residual.describe())

#ERROR FINAL
print('MAE: %.3f -- MSE: %.3f -- R2: %.3f' % (
	mean_absolute_error(df, model_fit.fittedvalues),
	mean_squared_error(df, model_fit.fittedvalues),
	r2_score(df, model_fit.fittedvalues)))