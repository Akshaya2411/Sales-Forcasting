from winreg import ExpandEnvironmentStrings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, date
import datetime as dt
from datetime import timedelta
import scipy.stats as scs
import warnings;
from sklearn.metrics import  r2_score
warnings.simplefilter('ignore')
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from  prophet import Prophet
from statsmodels.tsa.stattools import adfuller
import logging
logging.getLogger().setLevel(logging.ERROR)
from flask import Flask, request,jsonify
from flask_cors import CORS
import json
from io import BytesIO
import base64
app=Flask(__name__)
CORS(app)

@app.route("/dashboard",methods=['GET','POST'])
def Predict_Sales():
    input=request.data.decode()
    input=json.loads(input)
    ds=input[0]
    n=int(input[1])
    my_path ="/Users/Akshaya/Sales_prediction_App/src/assets/"
    df=pd.DataFrame(ds,columns=['Sno','InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'])
    df = df.iloc[1: , :]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'].str.strip(), dayfirst=True)
    df.set_index('InvoiceDate',inplace=True)
    df.to_csv('C:/Users/Akshaya/Sales_Prediction_App/src/assets/df.csv')
    print(df.head)
    print(df.info())
    (df.isnull().sum()/ df.shape[0]).sort_values(ascending=False)
    df_clean = df.copy()
    df_clean.dropna(axis=0, inplace=True)
    print(df_clean.head())
    (df_clean.isnull().sum()/ df_clean.shape[0]).sort_values(ascending=False)
    print(df_clean.head())
    df_clean['Quanity'] = pd.to_numeric(df_clean['Quantity'])
    df_clean = df_clean[df_clean.Quantity > 0]
    df_clean['CustomerID'] = df_clean['CustomerID'].astype('int64')
    df_clean['ActualSales'] = df_clean['Quantity'] * df_clean['UnitPrice']
    print(df_clean.head)
    df_clean["Year"] = df_clean.index.year
    df_clean["Quarter"] = df_clean.index.quarter
    df_clean["Month"] = df_clean.index.month
    df_clean["Week"] = df_clean.index.week
    df_clean["Weekday"] = df_clean.index.weekday
    df_clean["Day"] = df_clean.index.day
    df_clean["Dayofyear"] = df_clean.index.dayofyear
    df_clean["Date"] = pd.DatetimeIndex(df_clean.index).date
    df_clean['Weekend'] = 0
    print(df_clean.head())
    df_clean.to_csv('C:/Users/Akshaya/Sales_Prediction_App/src/assets/data.csv')
    df_clean.loc[(df_clean.Weekday == 5) | (df_clean.Weekday == 6), 'Weekend'] = 1
    grouped_features = ["Date", "Year", "Quarter","Month", "Week", "Weekday", "Dayofyear", "Day","StockCode"]
    daily_data = pd.DataFrame(df_clean.groupby(grouped_features).Quantity.sum(),columns=["Quantity"])
    daily_data["ActualSales"] = df_clean.groupby(grouped_features).ActualSales.sum()
    daily_data = daily_data.reset_index()
    daily_data.loc[:, ["Quantity", "ActualSales"]].describe()
    low_quantity = daily_data.Quantity.quantile(0.01)
    high_quantity = daily_data.Quantity.quantile(0.99)
    print((low_quantity, high_quantity))
    low_AmountSpent = daily_data.ActualSales.quantile(0.01)
    high_AmountSpent = daily_data.ActualSales.quantile(0.99)
    print((low_AmountSpent, high_AmountSpent))
    samples = daily_data.shape[0]
    daily_data = daily_data.loc[
    (daily_data.Quantity >= low_quantity) & (daily_data.Quantity <= high_quantity)]
    daily_data = daily_data.loc[
    (daily_data.ActualSales >= low_AmountSpent) & (daily_data.ActualSales <= high_AmountSpent)]
    samples = daily_data.shape[0]
    df_ts=daily_data.groupby('Date',as_index=False)['ActualSales'].sum()
    df_ts['Date'] = pd.to_datetime(df_ts['Date'], format="%Y-%m-%d")
    df_ts.index = pd.to_datetime(df_ts['Date'], format="%Y-%m-%d")
    df_ts.drop('Date', axis=1, inplace=True)
    df_ts = df_ts.reset_index()
    df_ts.columns = ['ds', 'y']
    print(df_ts.head())
    df_ts.index = pd.to_datetime(df_ts['ds'], format="%Y-%m-%d")
    df_ts.drop('ds', axis=1, inplace=True)
    ts_log = np.log(df_ts)
    ma = df_ts.rolling(12).mean()
    df_log = np.log(df_ts)
    ma_log = df_log.rolling(12).mean()
    df_sub = (df_log - ma_log).dropna()
    ma_sub = df_sub.rolling(12).mean()
    std_sub = df_sub.rolling(12).std()
    X_sub = df_sub.y.values
    result_sub = adfuller(X_sub)
    print('Augmented Dickey–Fuller')
    print('Statistical Test: {:.4f}'.format(result_sub[0]))
    print('P Value: {:.10f}'.format(result_sub[1]))
    print('Critical Values:')
    for key, value in result_sub[4].items():
      print('\t{}: {:.4f}'.format(key, value))
    df_diff = df_sub.diff(1)
    ma_diff = df_diff.rolling(12).mean()
    std_diff = df_diff.rolling(12).std()
    X_diff = df_diff.y.dropna().values
    result_diff = adfuller(X_diff)
    print('Augmented Dickey–Fuller')
    print('Statistical Test: {:.4f}'.format(result_sub[0]))
    print('P Value: {:.10f}'.format(result_sub[1]))
    print('Critical Values:')
    for key, value in result_sub[4].items():
      print('\t{}: {:.4f}'.format(key, value))
    df_log.reset_index(inplace=True)
    prediction_size = n
    print(n)
    if (n>=len(df_log)):
      train_df = df_log[:-int(len(df_log)*0.5)]
    else:
      train_df = df_log[:-prediction_size]
    m = Prophet()
    m.fit(train_df)
    future = m.make_future_dataframe(periods=prediction_size)
    forecast = m.predict(future)
    forecast.head()
    m.plot(forecast).savefig(my_path+'st_forecast.png')
    m.plot_components(forecast).savefig(my_path+'st_components.png')
    cmp_df = forecast.set_index('ds')[['yhat', 'yhat_lower','yhat_upper']].join(df_log.set_index('ds'))
    def calculate_forecast_errors(df_ts, prediction_size):
      df =df_ts.copy()
      df['e'] = df['y'] - df['yhat']
      df['p'] = 100 * df['e'] / df['y']
      predicted_part = df[-prediction_size:]
      error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))
      return{'MAPE': error_mean('p'), 'MAE': error_mean('e')}
    err=[]
    for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():
        err.append(err_value)
        print(err_name, err_value)
    mape=err[0]
    mae=err[1]
    se = np.square(cmp_df['y']-cmp_df['yhat'])
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    print(rmse)
    df_final = forecast[['ds', 'yhat']]
    df_final['ActualSales'] = df_log['y']
    df_final['ActualSales'] = np.exp(df_final['ActualSales'].values)
    df_final['PredictedSales'] = np.exp(df_final['yhat'].values)
    df_final.head(10)
    df_final.to_csv('C:/Users/Akshaya/Sales_Prediction_App/src/assets/result.csv')
    fig, ax = plt.subplots(figsize=(12,8))
    df_final['ActualSales'].plot(ax=ax, legend=('Actual Sales'))
    df_final['PredictedSales'].plot(ax=ax, color='r', legend={'Predicted Sales'})
    plt.savefig(my_path+'st_yhat.png')
    result = df_final.to_json(orient="index")
    img_data={}
    with open(my_path+'st_forecast.png',mode='rb') as file:
      st_forecast=file.read()
      img_data['st_forecast']=base64.encodebytes(st_forecast).decode('utf-8')
    with open(my_path+'st_components.png',mode='rb') as file:
      st_components=file.read()
      img_data['st_components']=base64.encodebytes(st_components).decode('utf-8')
    with open(my_path+'st_yhat.png',mode='rb') as file:
      st_yhat=file.read()
      img_data['st_yhat']=base64.encodebytes(st_yhat).decode('utf-8')
    
    return jsonify(result,img_data,mape,mae,rmse)


if __name__ =='main':
  app.run(debug=True)