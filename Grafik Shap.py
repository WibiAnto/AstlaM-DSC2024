import dash
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from main import PreProcessing, ModelDevelopment, SHAPValue, get_stream_data
from dash import dcc, html

app = dash.Dash(__name__)

# Initialize variables
categorical_cols = ['product_category', 'payment_method', 'transaction_status', 'device_type', 'location']
numerical_cols = ['product_amount', 'transaction_fee', 'cashback', 'loyalty_points']
preprocessing = PreProcessing()
model_development = ModelDevelopment()

# Read data
data_path = "./data/data.csv"
data = pd.read_csv(data_path)

preprocessing_data = preprocessing.extract_date_time(data=data)
preprocessing.label_encoder(data=data[categorical_cols])
encoded_data = preprocessing.transform_encoder(data=data[categorical_cols])
preprocessing_data = pd.concat([preprocessing_data, encoded_data, data[numerical_cols]], axis=1)
preprocessing.normalize(data=preprocessing_data)
preprocessing_data = preprocessing.transform_normalize(data=preprocessing_data)

model_development.set_model(data=preprocessing_data)
preprocessing_data["fraud_score"] = model_development.inference_model(data=preprocessing_data)
threshold = model_development.contamination * max(preprocessing_data["fraud_score"])
explaiable = SHAPValue(data=preprocessing_data[preprocessing_data.columns[:-1]])  
explaiable.set_explainer(model=model_development.model.predict)

# Initial empty figure
initial_fig = go.Figure()

# Layout
app.layout = html.Div([
    dcc.Graph(id='shap-graph', figure=initial_fig),
    dcc.Interval(
        id='interval-component',
        interval=10000,  # in milliseconds
        n_intervals=0
    )
])

# Initialize streaming data status
stream_data_status = None

@app.callback(
    Output('shap-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_shap_graph(n):
    global stream_data_status
    
    if stream_data_status is None:
        stream_data_status = get_stream_data(data=data)
    
    # Get next row of streaming data
    try:
        row = next(stream_data_status)
    except StopIteration:
        return go.Figure()  # Handle end of streaming data
    
    stream_data = preprocessing.extract_date_time(data=row)
    encoded_stream_data = preprocessing.transform_encoder(data=row[categorical_cols])
    stream_data = pd.concat([stream_data, encoded_stream_data, row[numerical_cols]], axis=1)
    stream_data = preprocessing.transform_normalize(data=stream_data)
    stream_data["fraud_score"] = model_development.inference_model(data=stream_data)
    stream_data["label"] = stream_data["fraud_score"].apply(lambda x: 1 if x >= threshold else 0)
    
    shap_value = explaiable.get_shap_value(data=stream_data[stream_data.columns[:-2]])
    shap_value["features"] = stream_data.columns[:-2].tolist()
    shap_value["abs_value"] = abs(shap_value.shap_value)
    shap_value.sort_values(by="abs_value",ascending = False, inplace = True)
    
    fig = go.Figure(data=[
        go.Bar(
            x=shap_value["abs_value"].head(5),
            y=shap_value["features"].head(5),
            orientation='h'
        )
    ])
    fig.update_layout(title='SHAP Values for Streaming Data',
                      xaxis_title='SHAP Value',
                      yaxis_title='Features')

    return fig

if __name__ == "__main__":
    app.run(debug=True)
