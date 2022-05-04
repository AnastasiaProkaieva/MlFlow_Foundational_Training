
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
  
  # If you do not have the File > Upload Data menu option, uncomment and run these lines to load the dataset.
  white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
  red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")
  red_wine['is_red'] = 1
  white_wine['is_red'] = 0
  data = pd.concat([red_wine, white_wine], axis=0)
  # Remove spaces from column names
  data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
  # Let's check the header of the input data 
  print(f"Dataset shape is {data.shape}")
  data.head()
  return data

def dataset_prep(data, label='quality', table2save="wine_data", SEED=54):
  """
  Preparing Dataset, calling test_train split.
  If any feature preprocessing required can be called inside
  """
  FEATURES = list(data.drop(columns=[label]))
  X, Y = data.loc[:,FEATURES], data.loc[:,label]

  # We are stratifying on Y (to assure the equal distribution of the Target)
  # Train is chosen 80% of the original dataset
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,
                                                      stratify=Y, random_state=SEED)
  if table2save:
    # Save test and train into delta 
    # write into Spark DF 
    sparkDF = spark.createDataFrame(data)
    sparkDF.write.format("delta").mode("overwrite").option('overwriteSchema', 'true').saveAsTable(table2save)
  return X_train, y_train, X_test, y_test



