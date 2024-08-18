import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc

import plotly.express as px
from plotly.express import bar
from plotly.subplots import make_subplots
from plotly import graph_objects as go

import pandas as pd

import pathlib



#%%
#APP SETUP
external_stylesheets = [dbc.themes.CYBORG]
# Initialize the Dash app
app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets
)

# Constants for styling
PLOT_BACKGROUND = 'rgba(0,0,0,0)'
PLOT_FONT_COLOR = 'white'
COMMON_STYLE ={
    "font-size": "1.25rem",
    "color": " white",
    "text-align": "left",
    "margin-top": "1rem",
    "margin-bottom": "1rem",
    "font-weight": "none",
    "font-family": "Times New Roman",
    "background-color": "transparent" 
}
#LOGO 

# PATHS
BASE_PATH = pathlib.Path(__file__).parent.resolve()
ART_PATH = BASE_PATH.joinpath("artifacts").resolve()


# DATA
Churn_data = pd.read_pickle(ART_PATH.joinpath('/Users/ellandalla/Desktop/Customer_Churn_Analysis-/venv/Data/churn_prediction_df.pkl'))

Churn_data.rename(columns ={'Churn': "Actual_Churn", 
                            'predict':"Predicted_Churn",
                            "No": "No_Churn_Rate",
                            "Yes": "Churn_Rate"}, inplace = True)	


# Define layout components    
# Container 
title = dbc.Row(
    dbc.Col([
        html.H1("Customer Churn Analysis Dashboard", className="text-center"),
        dcc.Markdown("""A dashboard identifying customers likely to churn, accompanied by exploratory analysis that provides insights into the dataset and offers recommendations to help prevent churn.""",
                     style = COMMON_STYLE),
       
    ]
    )
    )


figure = dbc.Row(
    dbc.Col(
        dcc.Graph(id='graph-churn_rate_by_tenure'),
        width=8
    )
)

figure_2 = dbc.Row(
    dbc.Col(
        dcc.Graph(id='graph-churn_rate_by_total_charges_Contract'),
        width=12
    )
)

figure_3 = dbc.Row(
    dbc.Col(
        dcc.Graph(id='graph-churn_rate_by_categorical_drivers'),
        width=12
    )
)

Recommendation_title = dbc.Row(
    dbc.Col([
        html.H2("Conclusion and Recommendations", className="text-center"),
    ])
)


Accordion = dbc.Row(
    dbc.Col(
        dbc.Accordion(
            [ dbc.AccordionItem(
                dcc.Markdown( 
                    """From the exploratory analysis and the review of feature importance from the Decision Tree Classifier, it is evident that customers with a tenure of 40 days or less and those who subscribe to monthly contracts are more likely to churn. Additional factors contributing to churn include:
                            * Demographics: Senior citizens and customers without dependents.
                            * Lack of Subscriptions: Customers not subscribing to add-on services such as device protection plans, online backups, and tech support.""",
                    style = COMMON_STYLE), 
                title="Conclusion",
                ),
            dbc.AccordionItem(
                dcc.Markdown("""
                            *  **Develop Comprehensive Contract Plans with Enhanced Add-On Services for Data and Voice Customers :**
                                * Create mid to long-term contract plans that include valuable add-on services to incentivize customers to commit for longer periods. Add-on services include:
                                    * Online backup and tech support for internet subscribers.
                                    * Device protection plans for voice customers.    
                                    * Enhanced  movie and TV streaming offerings though partnership with local services.
                            *  **Develop and Conduct Targeted Marketing Campaigns:** 
                                * Personalize communication strategies for customers without dependents and address the needs and concerns of senior Citizens who are likely to churn.
                                * Promote enhanced services developed in recommendation 1 and the benefits of long-term contracts. """,
        style = COMMON_STYLE),
                title="Recommendations",
                ),
            ],
            start_collapsed = True,
            flush = True,),
    )
)

button = dbc.Row(
    dbc.Col([
        html.P("Download the strategy to optimize retention for customers who are more likely than not to churn.", style = COMMON_STYLE),
        dbc.Button("Strategy", id="btn", color="primary", style = COMMON_STYLE, ),
        dcc.Download(id="download"),
    ],
        width=12,
    )
)

        
app.layout = dbc.Container([
    title,
    figure,
    figure_2,
    figure_3,
    Recommendation_title,
    Accordion,
    html.Br(),
    button
], fluid=True)


# CALLBACKS 
# Fig Data+
@app.callback(
    Output('graph-churn_rate_by_tenure', 'figure'),
    Input('graph-churn_rate_by_tenure', 'id')
)

def update_figure(input_id):
    df = Churn_data[Churn_data["Predicted_Churn"]== "Yes"].copy()

    fig = px.scatter(
        data_frame=df,
        x = 'tenure',
        y = 'Churn_Rate',
        color = 'Churn_Rate', 
        opacity=0.5, 
        color_continuous_scale='IceFire', 
        hover_name='customerID',
        hover_data=['Actual_Churn', 'Predicted_Churn', 'Churn_Rate'],
    ) \
        .update_layout(
            { 'title': 'Customer Tenure vs Churn Rate',
                'plot_bgcolor': PLOT_BACKGROUND,
                'paper_bgcolor':PLOT_BACKGROUND,
                'font_color': PLOT_FONT_COLOR,
                'height':700
            }
        ) \
        .update_traces(
            marker = dict(size = 12)
        )
    
    return fig

@app.callback(
    Output('graph-churn_rate_by_total_charges_Contract', 'figure'),
    Input('graph-churn_rate_by_total_charges_Contract', 'id')
)

def update_figure(input_id3):
    df = Churn_data[Churn_data["Predicted_Churn"] == "Yes"].copy()
    #contract_df = df[df["Contract", "TotalCharges", "Churn_Rate"]]
    # Create subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Month-to-Month', 'One Year', 'Two Years'))
    
    for i, contract in enumerate(['Month-to-month', 'One year', 'Two year']):
        contract_df = df[df['Contract'] == contract]
        fig.add_trace(
        go.Scatter(
            x=contract_df['TotalCharges'],
            y=contract_df['Churn_Rate'],
            mode='markers',
            marker=dict(color=contract_df['Churn_Rate'], colorscale='IceFire', showscale=(i==0)),
            text=contract_df['customerID'],
            name= contract),
        row=1, col= i+1)
    
    # Update x-axis and y-axis labels
    fig.update_xaxes(title_text='Total Charges')
    fig.update_yaxes(title_text='Churn Rate')

    # Update layout and display the figure
    fig.update_layout(title='Total Charges vs Churn Rate by Contract Type',
                  showlegend=True,
                  height=600, width=1000,
                  coloraxis=dict(colorscale='IceFire', cmin=0, cmax=1),
                  coloraxis_colorbar=dict(title='Churn Rate', yanchor='top', y=1, len=0.75),
                  plot_bgcolor = PLOT_BACKGROUND,
                  paper_bgcolor = PLOT_BACKGROUND,
                  font_color =  PLOT_FONT_COLOR,
                  
                  )
    
    return fig
    

@app.callback(
    Output('graph-churn_rate_by_categorical_drivers', 'figure'),
    Input('graph-churn_rate_by_categorical_drivers', 'id')
)

def update_categorical_drivers(input_id_2):
    """Visualize Categorical Data"""
    df = Churn_data[Churn_data["Predicted_Churn"] == "Yes"].copy()
    
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove columns that should not be plotted
    exclude_columns = ['customerID', 'Predicted_Churn', 'Actual_Churn']
    categorical_columns = [col for col in categorical_columns if col not in exclude_columns]
    
    # Create subplots dynamically based on number of categorical columns
    num_cols = len(categorical_columns)
    num_rows = (num_cols -1) // 4 + 1  # Adjust rows based on columns
     
    # Create subplots with shared y-axis
    fig = make_subplots(
        rows=num_rows,
        cols=4,
        subplot_titles= categorical_columns,
        shared_yaxes=True,
        vertical_spacing=0.1,
    )
    for i, col in enumerate(categorical_columns):
        # Calculate count and mean churn rate for each category
        count_df = df[col].value_counts().reset_index()
        count_df.columns = [col, 'Customer_Count']
        
        mean_df = df.groupby(col)['Churn_Rate'].mean().reset_index()
        mean_df.columns = [col, 'Avg_Churn_Rate']
        
        final_df = pd.merge(count_df, mean_df, on=col)
        
        # Calculate subplot indices (1-based)
        row = (i // 4) + 1
        col_num = (i % 4) + 1
    
    # Add traces (bar plots) to subplots
        fig.add_trace(
            go.Bar(
                x= final_df[col],
                y= final_df['Avg_Churn_Rate'],
                marker=dict(color=final_df['Avg_Churn_Rate'], colorscale= 'Bluered', showscale=False),
                text=final_df['Customer_Count'],
                name= col),
            row= row, col= col_num
        )
        
        # Update layout
        fig.update_layout(
            title='Avg Churn Rate by Categorical Drivers',
            height=1200,
            width=1200,
            showlegend=False,  # Legend may not be necessary for categorical plots
            coloraxis=dict(colorscale= 'Bluered', cmin=0, cmax=1),
            coloraxis_colorbar=dict(title='Avg_Churn_Rate', yanchor='top', y=1, len=0.75),
            plot_bgcolor = PLOT_BACKGROUND,
            paper_bgcolor = PLOT_BACKGROUND,
            font_color =  PLOT_FONT_COLOR,
        )
    return fig
        
        
@app.callback(
    Output("download", "data"), 
    Input("btn", "n_clicks"), 
    prevent_initial_call=True,
)

def download_strategy(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return dcc.send_file("/Users/ellandalla/Desktop/Customer_Churn_Analysis-/venv/Data/recommendation.csv")
if __name__ == '__main__':
    app.run_server(debug=True)

    
# %%





	