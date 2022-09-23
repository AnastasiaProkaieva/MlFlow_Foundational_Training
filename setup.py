# Databricks notebook source
dbutils.widgets.removeAll()
dbutils.widgets.dropdown("reset_all_data", "True", ["True", "False"])
dbutils.widgets.text("db_prefix", "churn_mlops", "Database Name")

# COMMAND ----------

import re

try:
  min_required_version = dbutils.widgets.get("min_dbr_version")
except:
  min_required_version = "10.4"

version_tag = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
version_search = re.search('^([0-9]*\.[0-9]*)', version_tag)
assert version_search, f"The Databricks version can't be extracted from {version_tag}, shouldn't happen, please correct the regex"
current_version = float(version_search.group(1))
assert float(current_version) >= float(min_required_version), f'The Databricks version of the cluster must be >= {min_required_version}. Current version detected: {current_version}'
assert "ml" in version_tag.lower(), f"The Databricks ML runtime must be used. Current version detected doesn't contain 'ml': {version_tag} "


spark.conf.set("spark.databricks.cloudFiles.schemaInference.sampleSize.numFiles", "10")
#spark.conf.set("spark.databricks.cloudFiles.schemaInference.enabled", "true")


current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
if current_user.rfind('@') > 0:
  current_user_no_at = current_user[:current_user.rfind('@')]
else:
  current_user_no_at = current_user
  
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)


db_prefix = dbutils.widgets.get("db_prefix")
dbName = db_prefix+"_"+current_user_no_at
cloud_storage_path = f"/Users/{current_user}/demo/{db_prefix}"
reset_all = dbutils.widgets.get("reset_all_data") == "true"

if reset_all:
  print(f"Resetting the DataBase, new database will have the name of {dbName}")
  spark.sql(f"DROP DATABASE IF EXISTS {dbName} CASCADE")
  dbutils.fs.rm(cloud_storage_path, True)

print(f"we are using {dbName} database")
print("Database will be in cloud_storage_path {}".format(cloud_storage_path))
spark.sql(f"""create database if not exists {dbName} LOCATION '{cloud_storage_path}/tables' """)
spark.sql(f"""USE {dbName}""")

# COMMAND ----------

print("HERE YOUR DATABASE is", dbName, " PLEASE COPY THIS INTO YOUR NEXT STEP if required \n")
print("HERE YOUR USERNAME is  ", current_user_no_at)
