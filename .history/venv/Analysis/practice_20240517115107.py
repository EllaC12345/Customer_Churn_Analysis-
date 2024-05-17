
#%%
import pandas as pd
# importing spark session
from pyspark.sql import SparkSession

# data visualization modules 
import matplotlib.pyplot as plt
import plotly.express as px 

# pyspark SQL functions 
from pyspark.sql.functions import col, when, count, udf

# pyspark data preprocessing modules
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder

# pyspark data modeling and model evaluation modules
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# %%

#Building Spark Session
spark = SparkSession.builder.appName("Customer_Churn_Prediction").getOrCreate()
spark

# reading the data
data = spark.read.csv("/Users/ellandalla/Desktop/Customer_Churn_Analysis-/venv/Data/dataset.csv", header=True, inferSchema=True)
data.show(5)

# print the schema of the data
data.printSchema()

# Get data dimensions
print((data.count(), len(data.columns)))
columns = data.columns
data.describe().show()
# %%

# Exploratory Data Analysis

## Create a list of columns and their data types

#obtain the data types of the columns

numerical_cols = [col for col, dtype in data.dtypes if dtype in ['int', 'double']]
numerical_cols

categorical_cols = [col for col, dtype in data.dtypes if dtype in ['string']]

categorical_cols

# Create a dataframe to store numerical and categorical data
num_df = data.select(numerical_cols).toPandas()
num_df

cat_df = data.select(categorical_cols).toPandas()
cat_df

# Plot histograms for numerical data
num_fig = plt.figure(figsize=(20, 10))
for i in range(len(numerical_cols)):
    fig.add_subplot(3, 3, i+1)
    plt.hist(num_df[numerical_cols[i]])
    plt.title(numerical_cols[i])

# investigating the tenure column for potential outliers
num_df.tenure.describe()

# plot a correlation matrix for the numerical data
num_corr = num_df.corr()
num_corr


# Categorical features exploratory analysis
#%%
categorical_cols = [col for col, dtype in data.dtypes if dtype in ['string']]

cat_df[]

cat_fig = plt.figure(figsize=(20, 10))
ax = fig.gca()
cat_df.hist(ax=ax, bins=30)
