import dash
import pandas as pd
from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import shap
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objs as go  # Importing Plotly for charts
from main import *  # Assuming your main file contains the required functions

# Setup the Dash app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
df = pd.DataFrame()  # Initialize an empty DataFrame

# Replace with your actual CSV path
categorical_cols = ['product_category', 'payment_method', 'transaction_status', 'device_type', 'location']
numerical_cols = ['product_amount','transaction_fee','cashback','loyalty_points']
preprocessing = PreProcessing()
model_development = ModelDevelopment()

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
     Input("interval-component", "n_intervals")],
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
            defaultColDef={"filter": True, "floatingFilter": True,  "wrapHeaderText": True, "autoHeaderHeight": True, "initialWidth": 200 },
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
            dbc.Col([html.Div(id="shap-bar-chart"), html.Div(id="fraud-score-chart")], md=9)
        ]),
        dbc.Row(dbc.Col(html.Div(id="grid-container")), className="my-4"),
        dcc.Interval(
        id='interval-component',
        interval=10000,  # Update every 1000 milliseconds (1 second)
        n_intervals=0  # Initial number of intervals
        )
    ], fluid=True
)

# Callback to update the data and SHAP values for the selected user
@app.callback(
    Output("store-selected", "data"),
    Input("interval-component", "n_intervals")
)
def update_store(user):
    global df  # Use the global variable
    try:
        for row in get_stream_data(data=data):
            stream_data = preprocessing.extract_date_time(data=row)
            encoded_stream_data = preprocessing.transform_encoder(data=row[categorical_cols])
            stream_data = pd.concat([stream_data, encoded_stream_data, row[numerical_cols]], axis=1)
            stream_data_not_normalize = stream_data.copy()

            now = datetime.datetime.now()
            stream_data_not_normalize['transaction_date'] = now.strftime("%Y-%m-%d %H:%M:%S")
            stream_data_not_normalize = pd.concat([row[['user_id','product_amount','payment_method','device_type','location','transaction_status']],stream_data_not_normalize], axis=1)
            stream_data = preprocessing.transform_normalize(data=stream_data)
            stream_data["fraud_score"] = model_development.inference_model(data=stream_data)
            stream_data["label"] = stream_data["fraud_score"].apply(lambda x: 1 if x >= threshold else 0)
            stream_data_not_normalize["fraud_score"] = stream_data["fraud_score"]
            stream_data_not_normalize["label"] = stream_data["label"]
            shap_value = explaiable.get_shap_value(data=stream_data[stream_data.columns[:-2]])
            df = pd.concat([df,stream_data_not_normalize], ignore_index=True) # Update stream_data from your main function
            print(df.to_dict())
            if True:
                break

    except StopIteration:
        return {"stream_data": [], "shap_value": []}  # Handle end of generator

# Function to plot SHAP bar plot and save it as an image
#def plot_shap_bar(shap_value):
#    # Create SHAP bar plot and save to file
#    shap.plots.bar(shap_value, show=False)
#    plt.savefig('./assets/shap_bar_plot.png', bbox_inches='tight')  # Save the plot to a file
#    plt.close()  # Close the plot to free up memory

# Callback to create SHAP bar chart using shap.plots.bar
#@app.callback(
#    Output("shap-bar-chart", "children"),
#    Input("store-selected", "data")
#)
#def make_shap_plot(data):
#    shap_value = data.get("shap_value", [])
#    if not shap_value:  # Check if shap_value is available
#        return dbc.Card([dcc.Markdown("No SHAP values available.")])
#    
#    # Generate SHAP bar plot and save as an image
#    plot_shap_bar(shap_value)
#
#    return dbc.Card([
#        dbc.CardHeader(html.H2("SHAP Values for Fraud Analysis")),
#        html.Img(src='/assets/shap_bar_plot.png', style={'width': '100%'})  # Load image in the dashboard
#    ])

# Callback to create Fraud Score chart
#@app.callback(
#    Output("fraud-score-chart", "children"),
#    Input("store-selected", "data")
#)
#def make_fraud_score_chart(data):
#    stream_data = data.get("stream_data", [])
#    if not stream_data:  # Check if data is available
#        return dbc.Card([dcc.Markdown("No fraud scores available.")])
#    
#    # Create a DataFrame from the stream_data for plotting
#    df = pd.DataFrame(stream_data)
#
#    fig = go.Figure(data=[go.Scatter(x=df.index, y=df["fraud_score"], mode='lines')])
#    
#    return dbc.Card([
#        dbc.CardHeader(html.H2("Fraud Score Over Time")),
#        dcc.Graph(figure=fig)
#    ])

if __name__ == "__main__":
    app.run_server(debug=True)
