# Databricks notebook source
# MAGIC %md
# MAGIC ## Monthly AutoML Retrain
# MAGIC <img src="https://github.com/aminenouira-db/images/blob/main/mlops-end2end-flow-7.png?raw=true" width="1200">

# COMMAND ----------

# MAGIC %md ## Monthly training job
# MAGIC 
# MAGIC We can programatically schedule a job to retrain our model, or retrain it based on an event if we realize that our model doesn't behave as expected.
# MAGIC 
# MAGIC This notebook should be run as a job. It'll call the Databricks Auto-ML API, get the best model and request a transition to Staging.

# COMMAND ----------



# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("feature_table_name", "telco_churn_features_ft", "Feature Table name")
dbutils.widgets.text("model_name", "telco_churn_ft", "Model Name to be register")
dbutils.widgets.text("db_prefix", "churn_mlops", "Database Prefix")
dbutils.widgets.text("user_name", "anastasia_prokaieva", "User Name")
dbutils.widgets.text("table_name", "telco_churn_ft", "Delta Table Name")

# COMMAND ----------

db_name = dbutils.widgets.get("db_prefix") + "_" + dbutils.widgets.get("user_name")
delta_table_name = dbutils.widgets.get("table_name") + "_" + dbutils.widgets.get("user_name")
fs_table_name = dbutils.widgets.get("feature_table_name") + "_" + dbutils.widgets.get("user_name")
model_name_registry = dbutils.widgets.get("model_name") + "_" + dbutils.widgets.get("user_name")
print(f"Your database is {db_name}, this is where your table will be store.\n")
print(f"We will work with 2 tables {delta_table_name} and {fs_table_name}\n")
print(f"Your model will be saved under the name {model_name_registry}\n")

# COMMAND ----------

# DBTITLE 1,Load Features
from databricks.feature_store import FeatureStoreClient
import pyspark.sql.functions as F
fs = FeatureStoreClient()
features = fs.read_table(f'{db_name}.{fs_table_name}')
display(features)

# COMMAND ----------

# DBTITLE 1,Run AutoML
import databricks.automl
model = databricks.automl.classify(features, 
                                   target_col = "Churn", # select your target DF
                                   exclude_columns=["customerID"], # userID is a useles here, then drop it
                                   max_trials = 5, # to make it run faster please limit the amount of models trained 
                                   data_dir= "dbfs:/tmp/", # where the output will be store, the tmp is often the best option
                                   timeout_minutes=5) 

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient()

df = mlflow.search_runs(
  experiment_ids = ['3804581652095129'], # HERE PLACE YOUR EXPERIMENT FROM AUTOML RUN
  filter_string = "status = 'FINISHED'"
)

best_run_id = df.sort_values("metrics.test_log_loss")["run_id"].values[0]
best_run_id

# COMMAND ----------

df.head()

# COMMAND ----------

# DBTITLE 1,Register the Best Run
run_id = best_run_id
model_uri = f"runs:/{run_id}/model"
client.set_tag(run_id, key='db_table', value=f'{db_name}.{fs_table_name}')

model_details = mlflow.register_model(model_uri, model_name_registry)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC #### Request Transition to Staging
# MAGIC 
# MAGIC <img style="float: right" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn_move_to_stating.gif">
# MAGIC 
# MAGIC Our model is now read! Let's request a transition to Staging. 
# MAGIC 
# MAGIC While this example is done using the API, we can also simply click on the Model Registry button.

# COMMAND ----------

# DBTITLE 1,Add Descriptions
model_version_details = client.get_model_version(name=model_name_registry, version=model_details.version)

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using autoML and automatically getting the best model."
)

# COMMAND ----------

# DBTITLE 1,Transition model stage helper
from mlflow.utils.rest_utils import http_request
import json

client = mlflow.tracking.client.MlflowClient()

host_creds = client._tracking_client.store.get_host_creds()
host = host_creds.host
token = host_creds.token

def mlflow_call_endpoint(endpoint, method, body='{}'):
  if method == 'GET':
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
  else:
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body))
  return response.json()

# COMMAND ----------

# DBTITLE 1,Request transition to Staging
# Transition request to staging
staging_request = {'name': model_name_registry, 'version': model_details.version, 'stage': 'Staging', 'archive_existing_versions': 'true'}
mlflow_call_endpoint('transition-requests/create', 'POST', json.dumps(staging_request))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Batch Inference 
# MAGIC When you trained your model you want to simply score it.
# MAGIC 
# MAGIC MLflow helps you generate code for batch or streaming inference.
# MAGIC 
# MAGIC - In the MLflow Model Registry, you can automatically generate a notebook
# MAGIC 
# MAGIC - In the MLflow Run page for your model, you can copy the generated code snippet for inference on pandas or Apache Spark DataFrames.

# COMMAND ----------

import mlflow
from pyspark.sql.functions import struct

model_uri = f"models:/{model_name_registry}/{model_details.version}"

# create spark user-defined function for model prediction
predict = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="string")

# COMMAND ----------

table = spark.read.table(f'{db_name}.{fs_table_name}')
output_df = table.withColumn("prediction", predict(struct(*table.columns)))
display(output_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Serving 
# MAGIC  > Please be aware that here we place this code as an example since we are currently in a private preview of our Serverless Serving
# MAGIC  
# MAGIC  
# MAGIC  
# MAGIC 
# MAGIC ##Deploying the model for real-time inferences
# MAGIC 
# MAGIC <img style="float: right; margin-left: 20px" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn_realtime_inference.gif" />
# MAGIC 
# MAGIC Our marketing team also needs to run inferences in real-time using REST api (send a customer ID and get back the inference).
# MAGIC 
# MAGIC While Feature store integration in real-time serving will come with Model Serving v2, you can deploy your Databricks Model in a single click.
# MAGIC 
# MAGIC Open the Model page and click on "Serving". It'll start your model behind a REST endpoint and you can start sending your HTTP requests!
# MAGIC 
# MAGIC ```
# MAGIC import requests
# MAGIC import os
# MAGIC import requests
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import json
# MAGIC 
# MAGIC 
# MAGIC token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
# MAGIC auth_header = {"Authorization" : "Bearer " + token}
# MAGIC 
# MAGIC # enabeling the Serving with API for v2
# MAGIC endpoint_path = "/mlflow/endpoints-v2/enable" 
# MAGIC payload = {
# MAGIC   "registered_model_name": model_name
# MAGIC }
# MAGIC 
# MAGIC host = "https://e2-demo-field-eng.cloud.databricks.com/api/2.0"
# MAGIC full_url = f"{host}{endpoint_path}"
# MAGIC response = requests.post(url = full_url, json = payload, headers = auth_header)
# MAGIC 
# MAGIC if response.status_code != 200:
# MAGIC   raise ValueError("Error making POST request to Mlflow API")
# MAGIC   
# MAGIC 
# MAGIC def create_tf_serving_json(data):
# MAGIC   return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}
# MAGIC 
# MAGIC def score_model(dataset):
# MAGIC   url = 'https://e2-demo-field-eng.cloud.databricks.com/model-endpoint/YOUR_MODEL_NAME/VERSION/invocations'
# MAGIC   headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
# MAGIC   ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
# MAGIC   data_json = json.dumps(ds_dict, allow_nan=True)
# MAGIC   response = requests.request(method='POST', headers=headers, url=url, data=data_json)
# MAGIC   if response.status_code != 200:
# MAGIC     raise Exception(f'Request failed with status {response.status_code}, {response.text}')
# MAGIC   return response.json()
# MAGIC 
# MAGIC # Make predictions
# MAGIC result = score_model(X_test) 
# MAGIC ```

# COMMAND ----------



# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>

# COMMAND ----------


