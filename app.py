from dash import callback, Dash, dcc, html, Input, Output
from main import get_stream_data, KaplanMeier, ModelDevelopment, PreProcessing, SHAPValue

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "E-wallet Predictive Fraud Detection"
streamed_data = pd.DataFrame()
shap_values = pd.DataFrame()
agged_data = pd.DataFrame(columns=["user_id", "current_n", "status", "survival_probability"])

categorical_cols = ['product_category', 'payment_method', 'transaction_status', 'device_type', 'location']
numerical_cols = ['product_amount','transaction_fee','cashback','loyalty_points']
data_path = "./data/data.csv"
data = pd.read_csv(filepath_or_buffer=data_path)

# Fit PreProcessing
preprocessing = PreProcessing()
preprocessing_data = preprocessing.extract_date_time(data=data)
preprocessing.label_encoder(data=data[categorical_cols])
encoded_data = preprocessing.transform_encoder(data=data[categorical_cols])
preprocessing_data = pd.concat([preprocessing_data, encoded_data, data[numerical_cols]], axis=1)
preprocessing.normalize(data=preprocessing_data)
preprocessing_data = preprocessing.transform_normalize(data=preprocessing_data)

# Fit or Load Model
model_development = ModelDevelopment()
preprocessing_data["fraud_score"] = model_development.inference_model(data=preprocessing_data)
threshold = model_development.contamination*max(preprocessing_data["fraud_score"])
explaiable = SHAPValue(data=preprocessing_data[preprocessing_data.columns[:-1]])  
explaiable.set_explainer(model=model_development.model.predict)

# Set Kaplan Meier
km = KaplanMeier(N=1000)

# Dashboard Layout
app.layout = dbc.Container(
    [
        dcc.Store(id="store-selected"),
        dbc.Row(
            html.H1(
                "E-wallet Predictive Fraud Detection",
            ),
            className="bg-primary text-white p-2 mb-4"
        )
        ,
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
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
                                ],
                                className="bg-light",
                            ),
                            className="mb-4"
                        ),
                        dbc.Accordion(
                            [
                                dbc.AccordionItem(
                                    dcc.Markdown(
                                        """
                                        This dashboard simulates fraud detection in e-wallet transactions.
                                        """
                                    ),
                                    title="About Project"
                                ),
                                dbc.AccordionItem(
                                    dcc.Markdown(
                                        """
                                        This dataset simulates transactions from a digital wallet platform similar to popular services like PayTm in India or Khalti in Nepal. It contains 5000 synthetic records of various financial transactions across multiple categories, providing a rich source for analysis of digital payment behaviors and trends.
                                        """
                                    ), 
                                    title="Data Source"
                                ),
                                dbc.AccordionItem(
                                    dcc.Markdown(
                                        """
                                        Tegar Ridwansyah (Team Leader) @tegarridwansyah \
                                        M. Ribhan Hadiyan @Ribhanhadyan \
                                        Wibi Anto @WibiAnto
                                        """
                                    ), 
                                    title="AstlaM"
                                )
                            ], start_collapsed=True)
                    ], 
                    md=3, 
                    style={
                        'background-color': '#f8f9fa',
                        'padding': '20px',
                        'border-right': '1px solid #ddd'
                    }
                ),
                dbc.Col(
                    [
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id='payment-method-chart', style={'height': '300px', 'width': '100%'}, animate=True), md=3),
                            dbc.Col(dcc.Graph(id='product-category-chart', style={'height': '300px', 'width': '100%'}, animate=True), md=3),
                            dbc.Col(dcc.Graph(id='graph-3', style={'height': '300px', 'width': '100%'}, animate=True), md=3),
                            dbc.Col(dcc.Graph(id='graph-4', style={'height': '300px', 'width': '100%'}, animate=True), md=3),
                        ], 
                        className="mb-4"
                    ),
                    dcc.Graph(id='shap-graph', style={'height': '500px'}, animate=True),
                    html.Div(id="fraud-score-chart", style={'margin-top': '20px'})
                    ], 
                    md=9
                )
            ],
            className="mb-5"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(id='survival-analysis-graph', style={'height': '500px'}),
                        html.Div(id="survival-analysis-chart", style={'margin-top': '20px'})
                    ], 
                    md=9
                ),
                dbc.Col(id='predict-fraud', md=3, style={'background-color': '#f8f9fa', 'padding': '20px'})
            ], 
            className="mb-5"
        ),
        dbc.Row(
            dbc.Col(
                html.Div(id="grid-container", style={'padding': '20px', 'background-color': '#ffffff', 'border': '1px solid #ddd'})
            ), 
            className="mb-4"
        ),
        dcc.Interval(
            id='interval-component',
            interval=10000,
            n_intervals=0
        ),
        dcc.Interval(
            id='interval-component-streaming',
            interval=3000,
            n_intervals=0
        ),
        html.Div(id="c")
    ],
    fluid=True,
    style={'padding': '40px', 'background-color': '#f0f2f5'}
)

csv_generator = get_stream_data(data=data)

@callback(
    Output(component_id="c", component_property="children"),
    Input(component_id="interval-component-streaming", component_property="n_intervals")
)
def stream_data(n_intervals):
    global csv_generator, streamed_data, shap_values
    new_row = next(csv_generator)
    normalized_row = preprocessing.extract_date_time(data=new_row)
    encoded_row = preprocessing.transform_encoder(data=new_row[categorical_cols])
    normalized_row = pd.concat([normalized_row, encoded_row, new_row[numerical_cols]], axis=1)
    normalized_row = preprocessing.transform_normalize(data=normalized_row)
    normalized_row["fraud_score"] = model_development.inference_model(data=normalized_row)
    normalized_row["label"] = normalized_row["fraud_score"].apply(lambda x: 1 if x >= threshold else 0)
        
    inserted_data = pd.concat([new_row, normalized_row[["fraud_score", "label"]]], axis=1)
    streamed_data = pd.concat([streamed_data, inserted_data])
    shap_value = explaiable.get_shap_value(data=normalized_row[normalized_row.columns[:-2]])
    shap_value["features"] = normalized_row.columns[:-2].tolist()
    shap_value["user_id"] = [new_row["user_id"].values[0]]*len(normalized_row.columns[:-2].tolist())
    shap_values = pd.concat([shap_values, shap_value])
    return ""

@callback(
    [
        Output(component_id="payment-method-chart", component_property="figure"),
        Output(component_id="product-category-chart", component_property="figure"),
        Output(component_id="graph-3", component_property="figure"),
        Output(component_id="graph-4", component_property="figure"),
        Output(component_id="shap-graph", component_property="figure"),
        Output(component_id="survival-analysis-graph", component_property="figure"),
        Output(component_id="predict-fraud", component_property="children"),
        Output(component_id="grid-container", component_property="children")
    ],
    [
        Input(component_id="user-dropdown", component_property="value"),
        Input(component_id="interval-component-streaming", component_property="n_intervals")
    ]
)
def update_transaction_activity(selected_user, n):
    global streamed_data, shap_values, agged_data
    try:
        user_data = streamed_data[streamed_data["user_id"] == selected_user].reset_index(drop=True).reset_index()
        
        payment_freq = user_data['payment_method'].value_counts().head(3).reset_index()
        payment_freq.columns = ['payment_method', 'frequency']
        fig_payment = {
            "data": [
                {
                    "x": payment_freq["payment_method"],
                    "y": payment_freq["frequency"],
                    "type": "bar"
                }
            ],
            "layout": {
                "title": "Top 3 Payment Method", 
                "margin": {
                    "l": 50,
                    "r": 10,
                    "t": 50,
                    "b": 30
                },
                "xaxis": {
                    "title": "Metode Pembayaran",
                },
                "yaxis": {
                    "title": "Frekuensi"
                },
                "template": "plotly_white"
            }
        }     

        category_freq = user_data['product_category'].value_counts().head(3).reset_index()
        category_freq.columns = ['product_category', 'frequency']
        fig_category = {
            "data": [
                {
                    "x": category_freq["product_category"],
                    "y": category_freq["frequency"],
                    "type": "bar"
                }
            ],
            "layout": {
                "title": "Top 3 Product Category",
                "margin": {
                    "l": 50,
                    "r": 10,
                    "t": 50,
                    "b": 30
                },
                "xaxis": {
                    "title": "Kategori Produk",
                },
                "yaxis": {
                    "title": "Frekuensi",
                },
                "template": "plotly_white"
            }
        }

        if len(user_data) > 20:
            user_data = user_data.tail(20)
        fig_amount = {
            "data": [
                {
                    "x": user_data["index"],
                    "y": user_data["product_amount"]/1000,
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Normal"
                },
                {
                    "x": user_data[user_data["label"] == 1]["index"],
                    "y": user_data[user_data["label"] == 1]["product_amount"]/1000,
                    "type": "scatter",
                    "mode": "markers",
                    "name": "fraud"
                },
            ],
            "layout": {
                "title": "Total Transaction",
                "margin": {
                    "l": 50,
                    "r": 10,
                    "t": 50,
                    "b": 30
                },
                "xaxis": {
                    "title": "Tanggal",
                },
                "yaxis": {
                    "title": "Total",
                },
                "legend": {
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1,
                    "xanchor": "top",
                    "x": 1
                },
                "template": "plotly_white"
            }
        }

        fig_fee = {
            "data": [
                {
                    "x": user_data["index"],
                    "y": user_data["transaction_fee"]/10,
                    "type": "scatter",
                    "mode": "lines",
                    "name": "fee"
                },
                {
                    "x": user_data[user_data["label"] == 1]["index"],
                    "y": user_data[user_data["label"] == 1]["transaction_fee"]/10,
                    "type": "scatter",
                    "mode": "markers",
                    "name": "fraud"
                },
            ],
            "layout": {
                "title": "Fee transaction",
                "margin": {
                    "l": 50,
                    "r": 10,
                    "t": 50,
                    "b": 30
                },
                "xaxis": {
                    "title": "Tanggal"
                },
                "yaxis": {
                    "title": "Biaya Admin",
                },
                "legend": {
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1,
                    "xanchor": "top",
                    "x": 1
                },
                "template": "plotly_white"
            }
        }

        user_shap = shap_values[shap_values["user_id"] == selected_user]
        user_shap = user_shap.groupby(by=["features"]).mean(numeric_only=True).sort_values(by=["shap_value"],ascending=False).head(5)
        fig_shap = {
            "data": [
                {
                    "x": user_shap.index,
                    "y": user_shap["shap_value"],
                    "type": "bar",
                }
            ],
            "layout": {
                "title": "The most influential fraud factors",
                "margin": {
                    "l": 50,
                    "r": 10,
                    "t": 50,
                    "b": 30
                },
                "xaxis": {
                    "title": "Features",
                },
                "yaxis": {
                    "title": "SHAP",
                    "range": [min(user_shap["shap_value"])-0.01, max(user_shap["shap_value"])+0.01]
                },
                "template": "plotly_white"
            }
        }

        user_survival = agged_data[agged_data["user_id"] == selected_user].tail(1)
        agg_data = pd.DataFrame()
        agg_data["user_id"] = [selected_user]
        if len(user_survival) > 0:
            agg_data["current_n"] = [int(user_survival["current_n"].values[0] - user_data.tail(1)["label"].values[0])]
        else:
            agg_data["current_n"] = [1000]
        agg_data["status"] = int(user_data.tail(1)["label"].values[0])
        agg_data["survival_probability"] = km.get_survival_prob(current_n=agg_data["current_n"].values[0], status=agg_data["status"].values[0])        
        agged_data = pd.concat([agged_data, agg_data])

        user_survival = agged_data[agged_data["user_id"] == selected_user].reset_index(drop=True).reset_index()
        fig_survival = {
            "data": [
                {
                    "x": user_survival["index"],
                    "y": user_survival["survival_probability"],
                    "type": "scatter",
                    "mode": "lines",
                },
            ],
            "layout": {
                "title": "Customer Fraud Analysis",
                "margin": {
                    "l": 50,
                    "r": 10,
                    "t": 50,
                    "b": 30
                },
                "xaxis": {
                    "title": "Tanggal"
                },
                "yaxis": {
                    "title": "Score",
                },
                "template": "plotly_white"
            }
        }
        recommendation_action = km.predict_future(durations=user_survival[user_survival["status"]==1]["survival_probability"], survival_probability=user_survival[user_survival["status"]==1]["survival_probability"])
        
        grid = dag.AgGrid(
            id="grid",
            rowData=user_data.to_dict("records"),
            columnDefs=[
                {"field": "transaction_date", "cellRenderer": "markdown", "initialWidth": 250, "pinned": "left"},
            ] + [{"field": "payment_method"},{"field": "device_type"},
                 {"field": "location"},{"field": "transaction_status"},
                 {"field": "fraud_score"},{"field": "label"}],
            defaultColDef={"filter": True, "floatingFilter": True, "wrapHeaderText": True, "autoHeaderHeight": True, "initialWidth": 200},
            dashGridOptions={},
            style={"height": 600, "width": "100%"}
        )

        return fig_payment, fig_category, fig_amount, fig_fee, fig_shap, fig_survival, recommendation_action, grid
    except (KeyError, ValueError):
        return dash.no_update

if __name__ == "__main__":
    app.run_server(debug=True)