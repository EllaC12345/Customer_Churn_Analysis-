
#%%
import pandas as pd
# importing spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# data visualization modules 
import matplotlib.pyplot as plt
import plotly.express as px 

# pyspark SQL functions 
from pyspark.sql.functions import * 

# pyspark data preprocessing modules
from pyspark.ml.feature import * 
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

# Create a list of unique values for each categorical column
categorical_cols = [col for col, dtype in data.dtypes if dtype in ['string']]
unique_values = {}
for col in categorical_cols:
    {col: cat_df[col].unique().tolist()}
    unique_values[col] = cat_df[col].unique().tolist()


# Group by categorical columns and count the number of occurences
grouped_dfs = {}

for col in categorical_cols:
    grouped_dfs[col] = cat_df.groupby(col)\
    .size()\
    .reset_index(name='count')
    
Print
for col in categorical_cols:
    print(f"Unique values in {col}:\n{unique_values[col]}\n")

for col, grouped_df in grouped_dfs.items():
    print(f"Grouped by {col}:\n{grouped_df}\n")



# count the number of null values in the data


for column in data.columns:
    data.select(count(when(col(column).isNull(), column))).show()
#data.select(count(when(col("Contract").isNull(), "Null"))).show()

type(data)


#%%
##Data Preprocessing
# Handling missing values using imputer
columns_with_missing_values = [column for column in data.columns if data.where(col(column).isNull()).count() > 0]
columns_with_missing_values

# Create an imputer object
imputer = Imputer(inputCols=columns_with_missing_values, outputCols=columns_with_missing_values).setStrategy("mean")
imputer = imputer.fit(data)
data = imputer.transform(data)

# Missing values Crosscheck
columns_with_missing_values_2 = [column for column in data.columns if data.where(col(column).isNull()).count() > 0]
columns_with_missing_values_2

## Removing outliers
# identifying outliers in the tenure column
data.select("*").where(col("tenure") > 100).show()
# Remove outliers
data = data.filter(data["tenure"]< 100)

# coutliers crosscheck
data.select("*").where(col("tenure") > 100).show()


#%%
## Feature Engineering - Numerical
#Creating a Vector Assembler
numerical_vector_assembler = VectorAssembler(inputCols=numerical_cols, outputCol="numerical_features_vector")
numerical_vector_assembler
data = numerical_vector_assembler.transform(data)
data.show(5)

#scaling the numerical features
scaler = StandardScaler(inputCol="numerical_features_vector", outputCol="scaled_numerical_features", withStd=True, withMean=True)

data = scaler.fit(data).transform(data)
data.show(5)

## Feature Engineering - Categorical

categorical_cols_indexed = [name + "_Indexed" for name in categorical_cols ]
categorical_cols_indexed
indexer = StringIndexer(inputCols = categorical_cols, outputCols = categorical_cols_indexed)
data = indexer.fit(data).transform(data)
data.show(5)

categorical_cols_indexed.remove("Churn_Indexed")
categorical_cols_indexed.remove("customerID_Indexed")
categorical_cols_indexed

categorical_vector_assembler = VectorAssembler(inputCols=categorical_cols_indexed, outputCol="categorical_features_vector")
data = categorical_vector_assembler.transform(data)
data.show(5)

final_vector_assembler = VectorAssembler(inputCols=["scaled_numerical_features", "categorical_features_vector"], outputCol="final_feature_vector")
data = final_vector_assembler.transform(data)
data.show

