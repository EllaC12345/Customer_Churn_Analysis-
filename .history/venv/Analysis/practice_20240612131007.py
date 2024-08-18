
#%%
import pandas as pd
# importing spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# data visualization modules 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 

# pyspark SQL functions 
from pyspark.sql.functions import * 

# pyspark data preprocessing modules
from pyspark.ml.feature import * 
from pyspark.sql.functions import count, when, col
# pyspark data modeling and model evaluation modules
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml import *

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

data.dtypes
# %%

data_df = data.toPandas()

def numeric_profile_data(data):
    """Pandas Profiling Function for Numeric Data

    Args:
        data (DataFrame): A data frame to profile

    Returns:
        DataFrame: A data frame with profiled data
    """
    numeric_data = data.select([col for col, dtype in data.dtypes if dtype in ['int', 'bigint', 'float', 'double']])
    profile_df = numeric_data.describe().toPandas().transpose()
    return profile_df
   
numeric_profile_data(data)
  

numeric_profile_data(data_df)

def category_profile_data(data):
    """Panda Profiling Function

    Args:
        data (DataFrame): A data frame to profile

    Returns:
        DataFrame: A data frame with profiled data
    """
    
    data = data.select_dtypes(include=['object'])
    profile_df = pd.concat([
        pd.DataFrame({'Dtype': data.dtypes}, index=data.columns),
            # Counts
            data.count().rename("Count"),
            data.isnull().sum().rename("NA Count"),
            data.nunique().rename("Count Unique"),
            # Stats
            data.mode().iloc[0].rename("Mode"),
        ],axis=1)
    return profile_df

category_profile_data(data_df)
data_df.columns
## Create a list of columns and their data types
data_df.dtypes
#obtain the data types of the columns
columns_with_datatypes = data.dtypes
numerical_cols = [col_name for col_name, col_type in columns_with_datatypes if col_type in ['int', 'bigint', 'float', 'double']]

categorical_cols = [col_name for col_name, col_type in columns_with_datatypes if col_type in ['string']]

categorical_cols


# Plot histograms for numerical data
num_fig = plt.figure(figsize=(20, 10))
for i in range(len(numerical_cols)):
    num_fig.add_subplot(3, 3, i+1)
    plt.hist(data_df[numerical_cols[i]])
    plt.title(numerical_cols[i])
    

# Spend Analysis 

# Investigating the Monthly Charges column for potential outliers

for i in range(len(numerical_cols)):
    px.box(data_df, y=numerical_cols[i], title=f"{numerical_cols[i]} Boxplot").show()



# Categorical features exploratory analysis
#%%

# Create a list of unique values for each categorical column
categorical_cols = [col for col, dtype in data.dtypes if dtype in ['string']]
unique_values = {}
for col in categorical_cols:
    {col: data_df[col].unique().tolist()}
    unique_values[col] = data_df[col].unique().tolist()


# Group by categorical columns and count the number of occurences
grouped_dfs = {}

for col in categorical_cols:
    grouped_dfs[col] = data_df.groupby(col)\
    .size()\
    .reset_index(name='count')
    

for col in categorical_cols:
    print(f"Unique values in {col}:\n{unique_values[col]}\n")

for col, grouped_df in grouped_dfs.items():
    print(f"Grouped by {col}:\n{grouped_df}\n")

# Visualizing the grouped data  
grouped_dfs.items()
for col, grouped_df in grouped_dfs.items():
    fig = px.bar(grouped_df, x=col, y="count", title=f"Customer Count by {col}", color="count")
    fig.show()

grouped_dfs




#%%
##Data Preprocessing
# Handling missing values using imputer
# Get column names and their corresponding data types


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
data.show(5)
data.select(["final_feature_vector", "Churn_Indexed"]).show(5)

#%%
#DecisionTreeClassifier
train, test = data.randomSplit([0.7, 0.3], seed = 42)
train.count(), test.count()

# Train the decision tree model
train.show(5)
dt = DecisionTreeClassifier(featuresCol="final_feature_vector", labelCol="Churn_Indexed", maxDepth=7)
pipeline = Pipeline(stages=[dt])
model = pipeline.fit(train)
?Pipeline
# Make predictions using test data
predictions_test = model.transform(test)
predictions_test.select("Churn", "Churn_Indexed", "prediction").show()

#%%
## Model Evaluation
evaluator = BinaryClassificationEvaluator(labelCol="Churn_Indexed")
auc_test = evaluator.evaluate(predictions_test, {evaluator.metricName: "areaUnderROC"})
auc_test

# evaluate the model using the training data
predictions_train = model.transform(train)
auc_train = evaluator.evaluate(predictions_train, {evaluator.metricName: "areaUnderROC"})
auc_train

def evaluate_dt(model_params):
    test_accuracies = []
    train_accuracies = []
    
    for maxDepth in model_params:
        # Train the decision tree model based on the maxDepth parameter
        dt = DecisionTreeClassifier(featuresCol="final_feature_vector", labelCol="Churn_Indexed", maxDepth=maxDepth)
        pipeline = Pipeline(stages=[dt])
        model = pipeline.fit(train)
        
        # Calculate the test error
        predictions_test = model.transform(test)
        evaluator = BinaryClassificationEvaluator(labelCol="Churn_Indexed")
        auc_test = evaluator.evaluate(predictions_test, {evaluator.metricName: "areaUnderROC"})
        # Append the test error to the test_accuracies list
        test_accuracies.append(auc_test)
        
        # Calculate the train error
        predictions_training = model.transform(train)
        evaluator = BinaryClassificationEvaluator(labelCol="Churn_Indexed")
        auc_training = evaluator.evaluate(predictions_training, {evaluator.metricName: "areaUnderROC"})
        train_accuracies.append(auc_training)
    return(test_accuracies, train_accuracies)

maxDepths = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
test_accs, train_accs = evaluate_dt(maxDepths)
df = pd.DataFrame(list(zip(maxDepths, test_accs, train_accs)), columns = ["maxDepth", "test_accuracy", "train_accuracy"], index= [numerical_cols + categorical_cols_indexed]
                  )

df


#%%
#Model Deployment
# How to reduce the churn rate
# Get the feature importances
dt_model = model.stages[-1]
feature_importance = dt_model.featureImportances
feature_importance
scores = []
for  index, importance in enumerate(feature_importance):
    score = [index, importance]
    scores.append(score)
        
#?model.featureImportances  
print(scores)
df = pd.DataFrame(scores, columns=[ "feature_number", "score"], index = categorical_cols_indexed + numerical_cols)
df
df_sorted = df.sort_values(by="score", ascending=False)
fig = px.bar(df_sorted, x=df_sorted.index, y="score", title="Feature Importance")
fig.update_layout(xaxis = {'categoryorder':'total descending'})

#%%
# lets create a Bar Chart to visualize the customer churn rate by tenure, by gender and  device protection plans,

df = data.groupBy("tenure", "Churn").count().toPandas()
df['tenure_quartile'] = pd.qcut(df['tenure'], q=4, labels=["Quart_1", "Quart_2", "Quart_3", "Quart_4"])
df_sorted = df.sort_values(by="tenure", ascending=True)
df_sorted 
# ReviewChurn Rate by Tenure Quartile
fig = px.bar(df_sorted, x="tenure_quartile", y="count", color="Churn", title="Customer Churn Rate by Tenure") 
fig.show()  
#import pyspark
#print(pyspark.__version__)
#print(nbformat.__version__)
#!pip install --upgrade nbformat
# %%
gender_df = data.groupBy('gender', 'Churn').count().toPandas()
gender_df
fig = px.bar(gender_df, x = "gender", y="count", color="Churn", title="Customer Churn Rate by Gender")
fig.show()


# Understanding the churn rate by device protection plan

device_protection_df = data.groupBy('DeviceProtection', 'Churn').count().toPandas()
pivot_df = device_protection_df\
    .groupby(['DeviceProtection', 'Churn'])\
    .agg({'count': 'sum'})\
    .reset_index()\
    .set_index('DeviceProtection')
pivot_df
    
    .plot(kind='bar', stacked=True, x='index', y = 'count', color = 'Churn', title="Customer Churn Rate by Device Protection Plan")
fig = px.bar(pivot_df, x = pivot_df.index, y="count", color="Churn", title="Customer Churn Rate by Device Protection Plan")
fig.show()