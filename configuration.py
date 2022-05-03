# Databricks notebook source
# MAGIC %md ## Import data
# MAGIC   
# MAGIC In this section, you download a dataset from the web and upload it to Databricks File System (DBFS).
# MAGIC 
# MAGIC 1. Navigate to https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/ and download both `winequality-red.csv` and `winequality-white.csv` to your local machine.
# MAGIC 
# MAGIC 1. From this Databricks notebook, select *File* > *Upload Data*, and drag these files to the drag-and-drop target to upload them to the Databricks File System (DBFS). 
# MAGIC 
# MAGIC     **Note**: if you don't have the *File* > *Upload Data* option, you can load the dataset from the Databricks example datasets. Uncomment and run the last two lines in the following cell.
# MAGIC 
# MAGIC 1. Click *Next*. Some auto-generated code to load the data appears. Select *pandas*, and copy the example code. 
# MAGIC 
# MAGIC 1. Create a new cell, then paste in the sample code. It will look similar to the code shown in the following cell. Make these changes:
# MAGIC   - Pass `sep=';'` to `pd.read_csv`
# MAGIC   - Change the variable names from `df1` and `df2` to `white_wine` and `red_wine`, as shown in the following cell.

# COMMAND ----------

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


# COMMAND ----------

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

user_name = user_name_set()
database_name = f'{user_name}_db'
# creating if were not a database with a table
print("Creating and setting the database if not exists")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")
spark.sql(f"USE {database_name}")

# COMMAND ----------


