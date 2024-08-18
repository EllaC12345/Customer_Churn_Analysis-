
#%%
import pandas as pd
# importing spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import col, count, when, isnan, isnull, mean, min, max

# data visualization modules 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 

# pyspark SQL functions 
from pyspark.sql.functions import * 

# pyspark data preprocessing modules
from pyspark.ml.feature import * 
from pyspark.sql.functions import count, when, col
from pyspark.sql import DataFrame
# pyspark data modeling and model evaluation modules
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml import *



#AutoML prediction
from pyspark.sql import SparkSession
import h2o
from ai.h2o.sparkling import H2OContext, H2OConf
from h2o.automl import H2OAutoML


#import Data
# %%
#initiate Spark Session
spark = SparkSession.builder \
    .appName("Read CSV to Spark DataFrame and Run H2O AutoML") \
    .getOrCreate()

    
# reading the data
final_data = spark.read.csv("/Users/ellandalla/Desktop/Customer_Churn_Analysis-/venv/Data/final_data.csv", header=True, inferSchema=True)
final_data.show(5)

data_df = final_data.toPandas()

# 1.0 H2O PREPARATION

# Initialize H2O

h2o.init(
    max_mem_size = 4,
    strict_version_check=False
)

# Convert to H2O Frame
data_h2o = h2o.H2OFrame(data_df)

data_h2o.describe()

data_h2o['Churn'] = data_h2o['Churn'].asfactor()

# Prep for AutoML

print(data_h2o.columns)
x_cols = [#'customerID', 
          'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
          'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
          'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
          'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
          'MonthlyCharges', 'TotalCharges', 
          
          ]
y_col = 'Churn'
#%%
# 2.0 RUN H2O AUTOML ----

# H2OAutoML
aml = H2OAutoML(
    nfolds= 5,
    max_runtime_secs= 3*60,
    seed = 123,
    exclude_algos= ['Deeplearning']  
)

aml.train(
    x = x_cols,
    y = y_col,
    training_frame= data_h2o
)

leaderboard_df = aml.leaderboard.as_data_frame()


#Save and Load the Model
model_h2o_stacked_ensemble = h2o.get_model(
    model_id = leaderboard_df['model_id'][0]
)

h2o.save_model(model_h2o_stacked_ensemble,
               path = '/Users/ellandalla/Desktop/Customer_Churn_Analysis-/venv/Models',
               filename= 'model_h2o_stacked_ensemble',
               force = True)

h2o.load_model('/Users/ellandalla/Desktop/Customer_Churn_Analysis-/venv/Models/model_h2o_stacked_ensemble')

h2o.__version__ 
h2o_version = '3.46.0.3'


#Prediction

predictions_h2o = model_h2o_stacked_ensemble.predict(data_h2o)

predictions_df = predictions_h2o.as_data_frame()

predictions_df 

final_prediction_df = pd.concat([data_df, predictions_df], axis = 1)

df = final_prediction_df




#Evaluate Plot Performance
model_performance  = model_h2o_stacked_ensemble.model_performance(data_h2o)
# %%
roc_plot = model_performance.plot(type = 'roc')

pr_plot = model_performance.plot(type = 'pr')

gain_lift_plot = model_performance.plot(type = 'gains_lift')




df.to_pickle('/Users/ellandalla/Desktop/Customer_Churn_Analysis-/venv/Data/churn_prediction_df.pkl')

#vizualization of potential churn customers


def visualize_categorical_data(data):
    """Visualize Categorical Data"""
    
    df = data.copy()
    
    categorical_columns = churn_prediction_df.select_dtypes(include=['object']).columns.tolist()
    
    categorical_columns.remove(['customerID', 'predict'])
    # Iterate over each categorical column and create a bar plot
    for col in categorical_columns:
        # Calculate the count for each category in the column
        count_df = churn_prediction_df[col].value_counts().reset_index()
        count_df.columns = [col, 'count']
        
        # Create the bar plot
        fig = px.bar(count_df, x=col, y='count', title=f"Customer Count by {col}", color='count')
        
        # Show the plot
        fig.show()


visualize_categorical_data(churn_prediction_df)


def visualize_numerical_data(data):
    """Visualize Numerical Data"""
 
    numerical_columns = churn_prediction_df.select_dtypes(include=['int', 'float']).columns.tolist()
    for col in ['No', 'Yes']:
        if col in numerical_columns:
            numerical_columns.remove(col)
    # Iterate over each numerical column and create a histogram
    for col in numerical_columns:
        # Create the histogram
        hist = px.histogram(churn_prediction_df, x=col, title=f"{col} Histogram")
        box = px.box(churn_prediction_df, y=col, title=f"{col} Boxplot")
        
        # Show the plot
        hist.show(), box.show()
    
visualize_numerical_data(churn_prediction_df)
       
       
# Determine the min probablity of churn

min_prob = df[df['predict'] == 'Yes']['Yes'].min()
# %%
