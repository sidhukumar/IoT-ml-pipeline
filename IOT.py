#!/usr/bin/env python
# coding: utf-8

# ### Installing the necessary Python packages.

# In[1]:

import os
import numpy as np
import pandas as pd
import time
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


# In[2]:
os.makedirs('./outputs', exist_ok=True)

iot = pd.read_csv(r'data/IOT-temp.csv')


# In[3]:


iot['noted_date'] = pd.to_datetime(iot['noted_date'])
iot['noted_date']=iot['noted_date'].apply(lambda x: time.mktime(x.timetuple()))


# In[4]:


#Spitting the Inside and outside temperatures
mask = iot['out/in'] == 'In'
iot_in = iot[mask]   #stores only those data points with inside temperatures
iot_out = iot[~mask] #stores only those data points with outside temperatures


# In[5]:


# Building the model for inside temperatures
X=iot_in[['noted_date']]
Y=iot_in[['temp']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=20)

lm = LinearRegression()
lm.fit(X_train,Y_train)

Y_predict = lm.predict(X_test)

filename = 'iot_in_model.pkl'
with open(os.path.join('./outputs/', filename), 'wb') as file:
    joblib.dump(lm,filename)


# ### Testing

# In[6]:


# Enter the date-time you want to see the predicted values for
date_str = "2020-04-16 12:02:00"

df = pd.to_datetime(date_str)
t = df.timetuple()
t = time.mktime(t)


# In[7]:


filename = 'iot_in_model.pkl'
model = joblib.load(filename)
y=model.predict([[t]])[0]
print(y)


# In[ ]:




