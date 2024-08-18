
# Importing Libraries
import pandas as pd


# Importing Data


Churn_data = pd.read_pickle('/Users/ellandalla/Desktop/Customer_Churn_Analysis-/venv/Data/churn_prediction_df.pkl')
Churn_data.rename(columns ={'Churn': "Actual_Churn", 
                            'predict':"Predicted_Churn",
                            "No": "No_Churn_Rate",
                            "Yes": "Churn_Rate"}, inplace = True)	
df = Churn_data.copy()



# Filtering definitions

def contract_label(contract):
    if contract == 'Month-to-month' or contract == 'One year':
        return 'Create and Promote mid to long term contracts'
    else:
        return 'Other'

def tech_service_label(row):
    if row['TechSupport'] == 'No' or row['OnlineBackup'] == 'No' or row['OnlineSecurity'] == 'No' or row['DeviceProtection'] == 'No':
        return 'Promote Tech Services'
    else:
        return 'Other'

def entertainment_label(row):
    if row['StreamingTV'] == 'Yes' or row['StreamingMovies'] == 'Yes':
        return 'Enhance and Promote Entertainment Services'
    else:
        return 'Other'

def communication_label(row):
    if row['SeniorCitizen'] == 'Yes':
        return 'Engage with senior citizens to better understand their needs and concerns.'
    elif row['Dependents'] == 'No':
        return 'Customize communication for customers with no dependents.'
    else:
        return 'Other'

# Assuming definitions to the dataframe columns

recommendation_df = df[(df['Predicted_Churn'] == "Yes") & (df['Churn_Rate'] > 0.50)] \
    .assign(contract_label=df['Contract'].apply(contract_label)) \
    .assign(tech_service_label=df.apply(lambda row: tech_service_label(row), axis=1)) \
    .assign(entertainment_label=df.apply(lambda row: entertainment_label(row), axis=1)) \
    .assign(communication_label=df.apply(lambda row: communication_label(row), axis=1))

# Print or use recommendation_df as needed
print(recommendation_df)

recommendation_df

for column in recommendation_df.columns:
    # Check if the column is not numeric (assuming non-numeric columns contain "others")
    if recommendation_df[column].dtype == 'object':
        # Check if any value in the column is "others"
        if (recommendation_df[column] == 'Other').any():
            # Replace "others" with "No action needed"
            recommendation_df.loc[recommendation_df[column] == 'Other', column] = 'No action needed'
            
            
recommendation_df['contract_label'].unique()


#save
recommendation_df.to_csv('/Users/ellandalla/Desktop/Customer_Churn_Analysis-/venv/Data/recommendation.csv')