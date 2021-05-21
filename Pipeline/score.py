import json
import os
import traceback

import numpy as np
import pandas as pd
import time
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

from azureml.core.model import Model


def init():
    global model
    filename = Model.get_model_path('IoT-model')
    model = joblib.load(filename+"/"+"iot_in_model.pkl")


def read():
    try:
        global t
        date_str = "2020-04-16 12:02:00"

        df = pd.to_datetime(date_str)
        t = df.timetuple()
        t = time.mktime(t)
    except Exception as err:
        traceback.print_exc()


def run(raw_data):
    data = np.array(json.loads(raw_data)[t])
    y = model.predict([[data]])[0]
    print(y)


# In[7]:



