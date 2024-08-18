
#%%

import os
import pandas as pd
import numpy as np
import pycaret
import pycaret.classification as clf
import mlflow.pyfunc


from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import col, count, when, isnan, isnull, mean, min, max



#Data Imports
#%%

spark = SparkSession.builder \
    .appName("Read CSV to Spark DataFrame and Run H2O AutoML") \
    .getOrCreate()

    
# reading the data
final_data = spark.read.csv("/Users/ellandalla/Desktop/Customer_Churn_Analysis-/venv/Data/final_data.csv", header=True, inferSchema=True)
final_data.show(5)

data_df = final_data.toPandas()
# %%

# Remove unnecessary columns
df = data_df\
    .drop(columns=['customerID'])
    
df.info()
df.head()

numeric_features = df.select_dtypes(include=[np.number]).columns.to_list()
categorical_features = df.select_dtypes(include=['object']).columns.to_list()

#%%
?clf.setup
clf_1 = clf.setup(
    data = df,
    target = 'Churn',
    train_size = 0.8,
    numeric_features = numeric_features,
    categorical_features = categorical_features,
    #ignore_features = None,
    preprocess = True,
    #normalize = True,
    #normalize_method = 'zscore',
    transformation = True,
    #remove_multicollinearity = True,
    #multicollinearity_threshold = 0.9,
    #bin_numeric_features = None,
    remove_outliers = True,
    outliers_threshold = 0.05,
    #create_clusters = False,
    #fix_imbalance = True,
    #fix_imbalance_method = 'smote',
    data_split_shuffle = True,
    data_split_stratify = True,
    fold_strategy = 'stratifiedkfold',
    fold = 5,
    #fold_shuffle = False,
    #fold_groups = None,
    n_jobs = -1,
    #use_gpu = False,
    session_id = 123,
    log_experiment = True,
    experiment_name = 'Churn_Analysis',
    log_plots = True,
    log_profile = True,
    log_data = True,
    verbose = True,
    profile = False,
    profile_kwargs = None,
    #display_container = True,
    #display=None,
    #display_format = None,
    )


# %%
