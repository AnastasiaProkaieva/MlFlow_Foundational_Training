# Databricks notebook source
# *****
# Importing Modules 
# *****
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

from pyspark.sql import SparkSession

# COMMAND ----------




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
  return userName

def read_wine_dataset():
  """
  If you have the File > Upload Data menu option, follow the instructions in the previous cell to upload the data from your local machine.
  The generated code, including the required edits described in the previous cell, is shown here for reference.

  In the following lines, replace <username> with your username.
  white_wine = pd.read_csv("/dbfs/FileStore/shared_uploads/<username>/winequality_white.csv", sep=';')
  red_wine = pd.read_csv("/dbfs/FileStore/shared_uploads/<username>/winequality_red.csv", sep=';')
  """
  import pandas as pd
  
  # If you do not have the File > Upload Data menu option, uncomment and run these lines to load the dataset.
  white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
  red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")
  red_wine['is_red'] = 1
  white_wine['is_red'] = 0
  data = pd.concat([red_wine, white_wine], axis=0)
  # Remove spaces from column names
  data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
  # Looks like quality scores are normally distributed between 3 and 9. 
  # Define a wine as high quality if it has quality >= 7.
  high_quality = (data.quality >= 7).astype(int)
  data.quality = high_quality
  # Let's check the header of the input data 
  print(f"Dataset shape is {data.shape}")
  return data

def print_input_shape(Xin, yin):
    print("Feature Table shape is ",Xin.shape)
    print('Target Label shape is', yin.shape)
    
def label_percnt(df, label = 1):
    return 100*(df == label).sum()/len(df)

def dataset_prep(data, label='quality', table2save="wine_data", SEED=54):
  """
  Preparing Dataset, calling test_train split.
  If any feature preprocessing required can be called inside
  """
  FEATURES = list(data.drop(columns=[label]))
  X, Y = data.loc[:,FEATURES], data.loc[:,label]

  # We are stratifying on Y (to assure the equal distribution of the Target)
  # Train is chosen 80% of the original dataset
  Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.20,
                                                      stratify=Y, random_state=SEED)
  if table2save:
    # Save test and train into delta 
    # write into Spark DF 
    sparkDF = spark.createDataFrame(data)
    sparkDF.write.format("delta").mode("overwrite").option('overwriteSchema', 'true').saveAsTable(table2save)
  
  print('We are splitting data into Train, Validation. Test sets ')    
  print('Train set')
  print_input_shape(Xtrain, ytrain)
  print('Test set')
  print_input_shape(Xtest, ytest)
  print('Verifying our label distribution')
  print(label_percnt(ytrain, label = 1))
  print(label_percnt(ytest, label = 1))
  return Xtrain, ytrain, Xtest, ytest


# COMMAND ----------



# COMMAND ----------

#def _prepare_spark() -> SparkSession:
#        if not spark:
#            return SparkSession.builder.getOrCreate()
#        else:
#            return spark
#spark = _prepare_spark()

# *****
# Defining some parameters 
# *****

