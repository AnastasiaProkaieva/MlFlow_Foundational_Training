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

# *****
# Defining some parameters 
# *****
# define a random hash 
uuid_num = uuid.uuid4().hex[:10]
# defining a user name 
