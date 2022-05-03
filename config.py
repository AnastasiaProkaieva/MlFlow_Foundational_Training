# Importing Modules 
import uuid
import os
import requests
import pandas as pd
import numpy as np


import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.tracking import MlflowClient


import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import cloudpickle
import time

from pyspark.sql.functions import struct
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from databricks import automl

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp

import xgboost as xgb



# define a random hash 
uuid_num = uuid.uuid4().hex[:10]
# defining a user name 
def getUsername() -> str:
    return spark.sql("SELECT current_user()").first()[0]
      
def user_name_set():
    try:
        username = spark.sql("SELECT current_user()").first()[0]
        userName = re.sub("[^a-zA-Z0-9]", "_", username.lower().split("@")[0])
        print(""" User name set to {}""".format(userName))
    except:
        userName = 'jon_snow'
        print("""We cannot access your user name, please check with your workspace administrator.
              \nUser Name set to {}""".format(userName))

