import dash
import pandas as pd
from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import shap

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.graph_objs as go  # Importing Plotly for charts
from main import *  # Assuming your main file contains the required functions
from lifelines import KaplanMeierFitter


# Setup the Dash app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
df = pd.DataFrame()  # Initialize an empty DataFrame
df_preprocessing = pd.DataFrame()

# Replace with your actual CSV path
categorical_cols = ['product_category', 'payment_method', 'transaction_status', 'device_type', 'location']
numerical_cols = ['product_amount','transaction_fee','cashback','loyalty_points']
preprocessing = PreProcessing()
model_development = ModelDevelopment()
date = random_date_transaction(30)

# Read data
data_path = "./data/data.csv"
data = pd.read_csv(filepath_or_buffer=data_path)

preprocessing_data = preprocessing.extract_date_time(data=data)
preprocessing.label_encoder(data=data[categorical_cols])
encoded_data = preprocessing.transform_encoder(data=data[categorical_cols])
preprocessing_data = pd.concat([preprocessing_data, encoded_data, data[numerical_cols]], axis=1)
preprocessing.normalize(data=preprocessing_data)
preprocessing_data = preprocessing.transform_normalize(data=preprocessing_data)

model_development.set_model(data=preprocessing_data)
preprocessing_data["fraud_score"] = model_development.inference_model(data=preprocessing_data)
threshold = model_development.contamination*max(preprocessing_data["fraud_score"])
explaiable = SHAPValue(data=preprocessing_data[preprocessing_data.columns[:-1]])  
explaiable.set_explainer(model=model_development.model.predict)

kmf = KaplanMeier(1000)

predict = ""


# Dropdown for selecting user
user_dropdown = html.Div(
    [
        dbc.Label("Select a user", html_for="user_dropdown"),
        dcc.Dropdown(
            id="user-dropdown",
            options=[{'label': user, 'value': user} for user in ['USER_00001', 'USER_00002', 'USER_00003', 'USER_00004', 'USER_00005']],
            value='USER_00001',
            clearable=False,
            maxHeight=600,
            optionHeight=50
        ),
    ], className="mb-4",
)

# Control panel layout
control_panel = dbc.Card(
    dbc.CardBody(
        [user_dropdown],
        className="bg-light",
    ),
    className="mb-4"
)

# Heading and description
heading = html.H1("E-wallet Transaction Fraud Detections", className="bg-secondary text-white p-2 mb-4")

# Accordion with information about data and project
about_card = dcc.Markdown(
    """
    This dashboard simulates fraud detection in e-wallet transactions.
    """
)

data_card = dcc.Markdown(
    """
    The dataset contains synthetic transactions data to simulate real-world scenarios.
    """
)

info = dbc.Accordion([
    dbc.AccordionItem(about_card, title="About Project"),
    dbc.AccordionItem(data_card, title="Data Source")
], start_collapsed=True)



# Callback to generate the transaction grid
@callback(
    Output("grid-container", "children"),
    [Input("user-dropdown", "value"),
     Input("interval-component", "n_intervals")]
)
def make_grid(user,interval):
    if not df.empty:
        filtered_df = df[df['user_id'] == user]
        grid = dag.AgGrid(
            id="grid",
            rowData=filtered_df.to_dict("records"),
            columnDefs=[
                {"field": "transaction_date", "cellRenderer": "markdown", "initialWidth": 250, "pinned": "left"},
            ] + [{"field": "payment_method"},{"field": "device_type"},
                 {"field": "location"},{"field": "transaction_status"},
                 {"field": "fraud_score"},{"field": "label"}],
            defaultColDef={"filter": True, "floatingFilter": True, "wrapHeaderText": True, "autoHeaderHeight": True, "initialWidth": 200},
            dashGridOptions={},
            style={"height": 600, "width": "100%"}
        )
        return grid
#[{"field": c} for c in numerical_cols]+

# Layout for the dashboard
app.layout = dbc.Container(
    [
        dcc.Store(id="store-selected", data={}),
        heading,
        dbc.Row([
            dbc.Col([control_panel, info], md=3),
            dbc.Col([dcc.Graph(id='shap-graph', figure=go.Figure()), html.Div(id="fraud-score-chart")], md=9)
        ]),
        dbc.Row([dbc.Col([dcc.Graph(id='survival-analysis-graph', figure=go.Figure()), html.Div(id="survival-analysis-chart")], md=9),
                 dbc.Col(id='predict-fraud')]),
        dbc.Row(dbc.Col(html.Div(id="grid-container")), className="my-4"),
        dcc.Interval(
        id='interval-component',
        interval=10000,  # Update every 1000 milliseconds (1 second)
        n_intervals=0  # Initial number of intervals
        ),
        dcc.Interval(
        id='interval-component-streaming',
        interval=1000,  # Update every 1000 milliseconds (1 second)
        n_intervals=0  # Initial number of intervals
        )
    ], fluid=True
)

# Callback to update the data and SHAP values for the selected user
@app.callback(
    Output("store-selected", "data"),
    Input("interval-component-streaming", "n_intervals")
)
def update_df(user):
    global df  # Use the global variable
    global df_preprocessing
    try:
        for row in get_stream_data(data=data):
            stream_data = preprocessing.extract_date_time(data=row)
            encoded_stream_data = preprocessing.transform_encoder(data=row[categorical_cols])
            stream_data = pd.concat([stream_data, encoded_stream_data, row[numerical_cols]], axis=1)
            stream_data_not_normalize = stream_data.copy()
            time = date.random_date()
            
            stream_data_not_normalize['transaction_date'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            stream_data_not_normalize = pd.concat([row[['user_id','product_amount','payment_method','device_type','location','transaction_status']],stream_data_not_normalize], axis=1)
            stream_data = preprocessing.transform_normalize(data=stream_data)
            stream_data["fraud_score"] = model_development.inference_model(data=stream_data)
            stream_data["label"] = stream_data["fraud_score"].apply(lambda x: 1 if x >= threshold else 0)
            stream_data["user_id"] = row['user_id']
            stream_data["date"] = time.strftime("%Y-%m-%d")
            df_preprocessing = pd.concat([df_preprocessing,stream_data], ignore_index=True)
            # pd.concat([df_preprocessing,stream_data]), stream_data.copy()
            stream_data_not_normalize["fraud_score"] = stream_data["fraud_score"]
            stream_data_not_normalize["label"] = stream_data["label"]
            #shap_value = explaiable.get_shap_value(data=stream_data[stream_data.columns[:-2]])
            df = pd.concat([df,stream_data_not_normalize], ignore_index=True) # Update stream_data from your main function
            #print(df.to_dict())
            if True:
                break

    except StopIteration:
        return {"stream_data": [], "shap_value": []}  # Handle end of generator

# Initialize streaming data status
stream_data_status = None

@app.callback(
    Output('shap-graph', 'figure'),
    [Input("user-dropdown", "value"),
     Input("interval-component", "n_intervals")]
)
def update_shap_graph(user,interval):
    filtered_df = df_preprocessing[df_preprocessing['user_id'] == user].tail(1)
    if (not filtered_df.empty):
        try:
            shap_value = explaiable.get_shap_value(data=filtered_df[filtered_df.columns[:-4]])
            shap_value["features"] = filtered_df.columns[:-4].tolist()
            shap_value["abs_value"] = abs(shap_value.shap_value)
            shap_value.sort_values(by="abs_value",ascending = False, inplace = True)
            #print(shap_value)
            fig = go.Figure(data=[
                go.Bar(
                    x=shap_value["abs_value"].head(5),
                    y=shap_value["features"].head(5),
                    orientation='h'
                )
            ])
            fig.update_layout(title=f'SHAP Values for Streaming Data {user}',
                              xaxis_title='SHAP Value',
                              yaxis_title='Features')

            return fig
        except StopIteration:
            return go.Figure()  # Handle end of streaming data
    


@app.callback(
    [Output('survival-analysis-graph', 'figure'),
     Output('predict-fraud','children')],
    [Input("user-dropdown", "value"),
     Input("interval-component", "n_intervals")]
)
def update_survival_analysis_graph(user,interval):
    global predict
    filtered_df = df_preprocessing[df_preprocessing['user_id'] == user].reset_index()
    if (not filtered_df.empty):
        try:
            grouped_year_month_day = filtered_df.groupby('date').sum(numeric_only=True).reset_index()
            status = grouped_year_month_day['label']
            probability = kmf.get_survival_prob(status)

            predict = kmf.predict_future(np.arange(0,len(probability)),probability)


            #if(user=='USER_00001'):
            #    probability = kmf_1.get_survival_prob(status)
            #elif(user=='USER_00002'):
            #    probability = kmf_2.get_survival_prob(status)
            #elif(user=='USER_00003'):
            #    probability = kmf_3.get_survival_prob(status)
            #elif(user=='USER_00004'):
            #    probability = kmf_4.get_survival_prob(status)
            #elif(user=='USER_00005'):
            #    kmf_5.get_survival_prob(status)
            #    probability = kmf_5.prob
            fig = go.Figure(data=go.Scatter(x=np.arange(0,len(probability)),
                                            y=probability),layout_yaxis_range=[min([0.8,min(probability)]),1.05]
            )
            fig.update_layout(title=f'Survival Analysis for Streaming Data {user}',
                              xaxis_title='Times',
                              yaxis_title='Probability',yaxis_range=[min([0.8,min(probability)]),1.05])

            return fig,predict
        except StopIteration:
            return go.Figure()  # Handle end of streaming data
    

if __name__ == "__main__":
    app.run_server(debug=True)
