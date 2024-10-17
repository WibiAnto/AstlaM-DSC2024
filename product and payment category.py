import dash
import pandas as pd
from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# Setup the Dash app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
data_path = "./data/data.csv"
data = pd.read_csv(filepath_or_buffer=data_path)

# Buat dropdown untuk user_id
user_ids = data['user_id'].unique()  # Ambil user_id yang unik

# Layout Dash
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='user-dropdown',
                options=[{'label': user_id, 'value': user_id} for user_id in user_ids],
                value=user_ids[0],  # Pilihan default
                multi=False  # Dropdown tunggal
            ),
            dcc.Graph(id='payment-method-chart'),
            dcc.Graph(id='product-category-chart')  # Grafik untuk kategori produk
        ])
    ])
])

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
        y=payment_freq['frequency'],
        marker_color='indigo'
    ))

    payment_chart.update_layout(
        title=f'Frekuensi Metode Pembayaran untuk User: {selected_user}',
        xaxis_title='Metode Pembayaran',
        yaxis_title='Frekuensi'
    )

    return payment_chart

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
        y=category_freq['frequency'],
        marker_color='indigo'
    ))

    category_chart.update_layout(
        title=f'Frekuensi Kategori Produk untuk User: {selected_user}',
        xaxis_title='Kategori Produk',
        yaxis_title='Frekuensi'
    )

    return category_chart

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
