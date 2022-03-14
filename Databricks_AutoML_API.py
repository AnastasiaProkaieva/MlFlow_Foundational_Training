# Databricks notebook source
# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - launch AutoML experiment from API
# MAGIC  - load best trial with spark_udf from AutoML and inference it
# MAGIC  - use Binary ClassificationEvaluator from SparkML 

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
# MAGIC 
# MAGIC For each lesson to execute correctly, please make sure to run the **`configuration`** cell at the start of each lesson.

# COMMAND ----------

# MAGIC %run ./configuration

# COMMAND ----------

# read the data
spark_df = spark.read.format("delta").table(f'{database_name}.wine_data')
train_df, test_df = spark_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

summary = automl.classify(train_df, target_col="quality", primary_metric="roc_auc", timeout_minutes=5, max_trials=20)

# COMMAND ----------

# let see the best trial 
print(summary.best_trial)

# COMMAND ----------

# MAGIC %md 
# MAGIC Example of how to inference the best automl run with spark 

# COMMAND ----------

model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"
predict = mlflow.pyfunc.spark_udf(spark, model_uri)
pred_df = test_df.withColumn("prediction", predict(*test_df.drop("price").columns))
display(pred_df)

# COMMAND ----------

clasify_evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="quality", metricName="areaUnderROC")
areaUnderROC = clasify_evaluator.evaluate(pred_df)
print(f"areaUnderROC on test dataset: {areaUnderROC:.3f}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>

# COMMAND ----------


