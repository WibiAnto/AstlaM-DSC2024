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
date = random_date_transaction(70)

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
    This dataset simulates transactions from a digital wallet platform similar to popular services like PayTm in India or Khalti in Nepal. It contains 5000 synthetic records of various financial transactions across multiple categories, providing a rich source for analysis of digital payment behaviors and trends.
    """
)

astlam_card = dcc.Markdown(
    """
    Tegar Ridwansyah (Team Leader) @tegarridwansyah \
    M. Ribhan Hadiyan @Ribhanhadyan \
    Wibi Anto @WibiAnto
    """
)

info = dbc.Accordion([
    dbc.AccordionItem(about_card, title="About Project"),
    dbc.AccordionItem(data_card, title="Data Source"),
    dbc.AccordionItem(astlam_card, title="AstlaM")
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
        # Store for selected data
        dcc.Store(id="store-selected", data={}),

        # Header
        heading,

        # Control panel and information on the left, main graphs on the right
        dbc.Row([
            dbc.Col([control_panel, info], md=3, style={'background-color': '#f8f9fa', 'padding': '20px', 'border-right': '1px solid #ddd'}),
            dbc.Col([
                # Row for the 4 small graphs
                dbc.Row([
                    dbc.Col(dcc.Graph(id='payment-method-chart', figure=go.Figure(), style={'height': '300px', 'width': '100%'}), md=3),
                    dbc.Col(dcc.Graph(id='product-category-chart', figure=go.Figure(), style={'height': '300px', 'width': '100%'}), md=3),
                    dbc.Col(dcc.Graph(id='graph-3', figure=go.Figure(), style={'height': '300px', 'width': '100%'}), md=3),
                    dbc.Col(dcc.Graph(id='graph-4', figure=go.Figure(), style={'height': '300px', 'width': '100%'}), md=3),
                ], className="mb-4"),

                # Main SHAP graph
                dcc.Graph(id='shap-graph', figure=go.Figure(), style={'height': '500px'}),

                # Fraud score chart
                html.Div(id="fraud-score-chart", style={'margin-top': '20px'})
            ], md=9)
        ], className="mb-5"),

        # Survival analysis row
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='survival-analysis-graph', figure=go.Figure(), style={'height': '500px'}),
                html.Div(id="survival-analysis-chart", style={'margin-top': '20px'})
            ], md=9),
            dbc.Col(id='predict-fraud', md=3, style={'background-color': '#f8f9fa', 'padding': '20px'})
        ], className="mb-5"),

        # Grid container
        dbc.Row(dbc.Col(html.Div(id="grid-container", style={'padding': '20px', 'background-color': '#ffffff', 'border': '1px solid #ddd'})), className="mb-4"),

        # Interval components for live updates
        dcc.Interval(
            id='interval-component',
            interval=10000,  # Update every 10 seconds
            n_intervals=0
        ),
        dcc.Interval(
            id='interval-component-streaming',
            interval=1000,  # Update every second
            n_intervals=0
        )
    ], fluid=True, style={'padding': '40px', 'background-color': '#f0f2f5'}
)


# Callback untuk memperbarui grafik kategori produk
@callback(
    Output('product-category-chart', 'figure'),
    Input('user-dropdown', 'value')
)
def update_product_category_chart(selected_user):
    # Filter data berdasarkan user_id yang dipilih
    user_data = data[data['user_id'] == selected_user]

    # Hitung frekuensi kategori produk
    category_freq = user_data['product_category'].value_counts().head(5).reset_index()
    category_freq.columns = ['product_category', 'frequency']

    # Buat grafik bar untuk frekuensi kategori produk
    category_chart = go.Figure()
    category_chart.add_trace(go.Bar(
        x=category_freq['product_category'],
        y=category_freq['frequency']
    ))

    category_chart.update_layout(
        #title=f'Frekuensi Kategori Produk untuk User: {selected_user}',
        xaxis_title='Kategori Produk',
        yaxis_title='Frekuensi'
    )

    return category_chart

# Callback untuk memperbarui grafik metode pembayaran
@callback(
    Output('payment-method-chart', 'figure'),
    Input('user-dropdown', 'value')
)
def update_payment_chart(selected_user):
    # Filter data berdasarkan user_id yang dipilih
    user_data = data[data['user_id'] == selected_user]

    # Hitung frekuensi metode pembayaran
    payment_freq = user_data['payment_method'].value_counts().head(5).reset_index()
    payment_freq.columns = ['payment_method', 'frequency']

    # Buat grafik bar untuk frekuensi metode pembayaran
    payment_chart = go.Figure()
    payment_chart.add_trace(go.Bar(
        x=payment_freq['payment_method'],
        y=payment_freq['frequency']
    ))

    payment_chart.update_layout(
        #title=f'Frekuensi Metode Pembayaran untuk User: {selected_user}',
        xaxis_title='Metode Pembayaran',
        yaxis_title='Frekuensi'
    )

    return payment_chart

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
