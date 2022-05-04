# Databricks notebook source
# MAGIC %md # Training machine learning models on tabular data: an end-to-end example
# MAGIC 
# MAGIC ### ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC - set a User name and create a database
# MAGIC - Import data from your local machine into the Databricks File System (DBFS)
# MAGIC - Visualize the data using Seaborn and matplotlib
# MAGIC - Create custom PyFunc model 
# MAGIC - Log model into MlFLow
# MAGIC - Register your first model version into MlFlow

# COMMAND ----------

# MAGIC %md 
# MAGIC In this example, you build a model to predict the quality of Portugese "Vinho Verde" wine based on the wine's physicochemical properties. 
# MAGIC 
# MAGIC The example uses a dataset from the UCI Machine Learning Repository, presented in [*Modeling wine preferences by data mining from physicochemical properties*](https://www.sciencedirect.com/science/article/pii/S0167923609001377?via%3Dihub) [Cortez et al., 2009].
# MAGIC 
# MAGIC #### Requirements
# MAGIC This notebook requires Databricks Runtime for Machine Learning 10.4 LTS or above.  

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Classroom-Setup
# MAGIC ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)
# MAGIC 
# MAGIC For each lesson to execute correctly, please make sure to run the **`configuration`** cell  or ro import everything from the **`config`** file at the start of each lesson.

# COMMAND ----------

from config import *

# COMMAND ----------

#%run ./configuration

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Create a database where all Delta Tables will be stored

# COMMAND ----------

user_name = user_name_set()
database_name = f'{user_name}_db'
# creating if were not a database with a table
print("Creating and setting the database if not exists")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")
spark.sql(f"USE {database_name}")

# COMMAND ----------

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

# MAGIC %md Merge the two DataFrames into a single dataset, with a new binary feature "is_red" that indicates whether the wine is red or white.

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Visualize data
# MAGIC 
# MAGIC Before training a model, explore the dataset using Seaborn and Matplotlib.

# COMMAND ----------

# MAGIC %md Plot a histogram of the dependent variable, quality.

# COMMAND ----------

import seaborn as sns
sns.distplot(data.quality, kde=False)

# COMMAND ----------

# MAGIC %md Looks like quality scores are normally distributed between 3 and 9. 
# MAGIC 
# MAGIC Define a wine as high quality if it has quality >= 7.

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

# MAGIC %md Box plots are useful in noticing correlations between features and a binary label.

# COMMAND ----------

import matplotlib.pyplot as plt

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue # Box plots cannot be used on indicator variables
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0

# COMMAND ----------

# MAGIC %md In the above box plots, a few variables stand out as good univariate predictors of quality. 
# MAGIC 
# MAGIC - In the alcohol box plot, the median alcohol content of high quality wines is greater than even the 75th quantile of low quality wines. High alcohol content is correlated with quality.
# MAGIC - In the density box plot, low quality wines have a greater density than high quality wines. Density is inversely correlated with quality.

# COMMAND ----------

# MAGIC %md ## Preprocess data
# MAGIC Prior to training a model, check for missing values and split the data into training and validation sets.

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md There are no missing values.

# COMMAND ----------

TARGET = 'quality'
X_train, y_train, X_test, y_test = dataset_prep(data, label=TARGET, table2save="wine_dataset")

# COMMAND ----------

display(spark.read.table("wine_dataset"))

# COMMAND ----------

# MAGIC %sql 
# MAGIC show tables

# COMMAND ----------

# MAGIC %sql 
# MAGIC describe history wine_dataset

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's explore our dataset with the Databricks Data Profiler 

# COMMAND ----------

display(spark.read.table("wine_dataset"))

# COMMAND ----------

# MAGIC %md ## Build a baseline model
# MAGIC This task seems well suited to a random forest classifier, since the output is binary and there may be interactions between multiple variables.
# MAGIC 
# MAGIC The following code builds a simple classifier using scikit-learn. It uses MLflow to keep track of the model accuracy, and to save the model for later use.

# COMMAND ----------

conda_env = mlflow.pyfunc.get_default_conda_env()
print(conda_env)

# COMMAND ----------

# The predict method of sklearn's RandomForestClassifier returns a binary classification (0 or 1). 
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to each class. 

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]
  
# mlflow.start_run creates a new MLflow run to track the performance of this model. 
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.

params_rf = {
  'n_estimators': 50,
  'max_depth':7,
  'random_state':np.random.RandomState(123)
            }

# To do allow users to use previously created experiments 
# set an experiment if does not exist 

# COMMAND ----------

conda_env

# COMMAND ----------


with mlflow.start_run(run_name=f'untuned_random_forest_{user_name}'):
  
  model = RandomForestClassifier(**params_rf)
  model.fit(X_train, y_train)
  # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  # Logging parameters used to train our model 
  mlflow.log_params(params_rf)
  # Use the area under the ROC curve as a metric.
  mlflow.log_metric('auc', auc_score)

  # Wrap the model 
  wrappedModel = SklearnModelWrapper(model)
  # Log the model with a signature that defines the schema of the model's inputs and outputs. 
  # When the model is deployed, this signature will be used to validate inputs.
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  # MLflow contains utilities to create a conda environment used to serve models.
  # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
  # log your model into MlFlow 
  mlflow.pyfunc.log_model(f"random_forest_model_{user_name}", python_model=wrappedModel, conda_env=conda_env, signature=signature)
  
  # Log metrics for the train and test set
  mdl_train_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_train, y_train, prefix="train_")
  mdl_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")
  # Display the logged metrics
  mdl_train_metrics = {k.replace("train_", ""): v for k, v in mdl_train_metrics.items()}
  mdl_test_metrics = {k.replace("test_", ""): v for k, v in mdl_test_metrics.items()}
  display(pd.DataFrame([mdl_train_metrics, mdl_test_metrics], index=["train", "test"]))
  

# COMMAND ----------

# MAGIC %md 
# MAGIC to read more about mlflow 

# COMMAND ----------



# COMMAND ----------

model = RandomForestClassifier(**params_rf)
# construct an evaluation dataset from the test set
eval_data = X_test
eval_data["target"] = y_test

with mlflow.start_run(run_name=f'untuned_random_forest_{user_name}'):
  
  model.fit(X_train, y_train)
  # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  # Logging parameters used to train our model 
  mlflow.log_params(params_rf)
  # Use the area under the ROC curve as a metric.
  mlflow.log_metric('auc', auc_score)
 
  model_info = mlflow.sklearn.log_model(model, "sklearn_rf_model")
  result = mlflow.evaluate(
       model_info.model_uri,
       eval_data,
       targets='target',
       model_type="classifier",
       dataset_name="wine_dataset",
       evaluators=["default"],
   )

# COMMAND ----------

# MAGIC %md Examine the learned feature importances output by the model as a sanity-check.

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md As illustrated by the boxplots shown previously, both alcohol and density are important in predicting quality.

# COMMAND ----------

# MAGIC %md You logged the Area Under the ROC Curve (AUC) to MLflow. Click **Experiment** at the upper right to display the Experiment Runs sidebar. 
# MAGIC 
# MAGIC The model achieved an AUC of 0.89. 
# MAGIC 
# MAGIC A random classifier would have an AUC of 0.5, and higher AUC values are better. For more information, see [Receiver Operating Characteristic Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).

# COMMAND ----------

# MAGIC %md #### Register the model in MLflow Model Registry
# MAGIC 
# MAGIC By registering this model in Model Registry, you can easily reference the model from anywhere within Databricks.
# MAGIC 
# MAGIC The following section shows how to do this programmatically, but you can also register a model using the UI. See "Create or register a model using the UI" ([AWS](https://docs.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html#create-or-register-a-model-using-the-ui)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/manage-model-lifecycle/index#create-or-register-a-model-using-the-ui)|[GCP](https://docs.gcp.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html#create-or-register-a-model-using-the-ui)).

# COMMAND ----------

run_id = mlflow.search_runs(filter_string=f'tags.mlflow.runName = "untuned_random_forest_{user_name}"').iloc[0].run_id

# COMMAND ----------

# If you see the error "PERMISSION_DENIED: User does not have any permission level assigned to the registered model", 
# the cause may be that a model already exists with the name "wine_quality". Try using a different name.
model_name = f"wine_quality_{user_name}_{uuid_num}"
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model_{user_name}", model_name)

# Registering the model takes a few seconds, so add a small delay
time.sleep(15)

# COMMAND ----------

# MAGIC %md You should now see the model in the Models page. To display the Models page, click the Models icon in the left sidebar. 
# MAGIC 
# MAGIC Next, transition this model to production and load it into this notebook from Model Registry.

# COMMAND ----------

print('Your registered model name is : {}'.format(model_name))

# COMMAND ----------

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------

# MAGIC %md The Models page now shows the model version in stage "Production".
# MAGIC 
# MAGIC You can now refer to the model using the path "models:/wine_quality/production".

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# Sanity-check: This should match the AUC logged by MLflow
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
