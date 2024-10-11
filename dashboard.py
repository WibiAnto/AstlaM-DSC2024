# Plotly's Figure Friday challenge. See more info here: https://community.plotly.com/t/figure-friday-2024-week-32/86401
import dash
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, callback, Patch
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.graph_objects as go

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

df = pd.read_csv('https://raw.githubusercontent.com/plotly/Figure-Friday/main/2024/week-32/irish-pay-gap.csv')
df['Report Link'] = df['Report Link'].apply(lambda x: f'[Report]({x})')
df['Company'] = df.apply(lambda row: f'[{row["Company Name"]}]({row["Company Site"]})', axis=1)
df.rename(columns={'Q1 Men': 'Q1 Male'}, inplace=True)

numeric_columns = [
   'Mean Hourly Gap', 'Median Hourly Gap', 'Mean Bonus Gap', 'Median Bonus Gap', 'Mean Hourly Gap Part Time',
   'Median Hourly Gap Part Time', 'Mean Hourly Gap Part Temp', 'Median Hourly Gap Part Temp', 'Percentage Bonus Paid Female',
   'Percentage Bonus Paid Male', 'Percentage BIK Paid Female', 'Percentage BIK Paid Male', 'Q1 Female', 'Q1 Male', 'Q2 Female',
   'Q2 Male', 'Q3 Female', 'Q3 Male', 'Q4 Female', 'Q4 Male', 'Percentage Employees Female', 'Percentage Employees Male'
]

company_dropdown = html.Div(
    [
        dbc.Label("Select a Company", html_for="company_dropdown"),
        dcc.Dropdown(
            id="company-dropdown",
            options=sorted(df["Company Name"].unique()),
            value='Ryanair',
            clearable=False,
            maxHeight=600,
            optionHeight=50
        ),
    ],  className="mb-4",
)

year_radio = html.Div(
    [
        dbc.Label("Select Year", html_for="date-checklist"),
        dbc.RadioItems(
            options=[2023, 2022],
            value=2023,
            id="year-radio",
        ),
    ],
    className="mb-4",
)

control_panel = dbc.Card(
    dbc.CardBody(
        [year_radio, company_dropdown ],
        className="bg-light",
    ),
    className="mb-4"
)

heading = html.H1("Ireland Gender Pay Gap Analysis",className="bg-secondary text-white p-2 mb-4")

about_card = dcc.Markdown(
    """
    The gender pay gap does not measure equal pay, instead it measures the difference between men and
    women's average and median hourly pay.  Equal pay, on the other hand, is the legal obligation under the Employment
     Equality Acts that requires  employers to give men and women equal pay if they are employed to do equal work. 
     
    Note that there is no equivalent reporting requirement in the US. Refer to this [US Department of Labour brief](https://www.dol.gov/sites/dolgov/files/WB/equalpay/WB_issuebrief-undstg-wage-gap-v1.pdf)
     which notes that "regardless of the gender composition of jobs, women tend to be paid less on average than men in the
     same occupation even when working full time."
    """)

data_card = dcc.Markdown(
    """
    Starting from 2022, Gender Pay Gap Reporting is a regulatory requirement that mandates employers in Ireland with
     more than 250 employees to publish information on their gender pay gap.
     
     [Data source](https://paygap.ie/)
     
     [Data source GitHub](https://github.com/zenbuffy/irishGenderPayGap/tree/main)
     
     This site was created for Plotly's Figure Friday challenge. For additional data visualizations of this dataset and
      to join the conversation, visit the [Plotly Community Forum](https://community.plotly.com/t/figure-friday-2024-week-32/86401)
    """
)

info = dbc.Accordion([
    dbc.AccordionItem(about_card, title="About Gender Pay Gap", ),
    dbc.AccordionItem(data_card, title="Data Source")
],  start_collapsed=True)

def make_grid():
    grid = dag.AgGrid(
        id="grid",
        rowData=df.to_dict("records"),
        columnDefs=[
          {"field": "Company", "cellRenderer": "markdown", "linkTarget": "_blank",  "initialWidth":250, "pinned": "left" },
          {"field": "Report Link", "cellRenderer": "markdown", "linkTarget": "_blank", "floatingFilter": False},
          {"field": "Report Year" }] +
        [{"field": c} for c in numeric_columns],
        defaultColDef={"filter": True, "floatingFilter": True,  "wrapHeaderText": True, "autoHeaderHeight": True, "initialWidth": 125 },
        dashGridOptions={},
        filterModel={'Report Year': {'filterType': 'number', 'type': 'equals', 'filter': 2023}},
        rowClassRules = {"bg-secondary text-dark bg-opacity-25": "params.node.rowPinned === 'top' | params.node.rowPinned === 'bottom'"},
        style={"height": 600, "width": "100%"}
    )
    return grid


app.layout = dbc.Container(
    [
        dcc.Store(id="store-selected", data={}),
        heading,
        dbc.Row([
            dbc.Col([control_panel, info], md=3),
            dbc.Col(
                [
                    dcc.Markdown(id="title"),
                    dbc.Row([dbc.Col(html.Div(id="paygap-card")), dbc.Col( html.Div(id="bonusgap-card"))]),
                    html.Div(id="bar-chart-card", className="mt-4"),
                ],  md=9
            ),
        ]),
        dbc.Row(dbc.Col( make_grid()), className="my-4")
    ],
    fluid=True,
)


@callback(
    Output("grid", "dashGridOptions"),
    Output("store-selected", "data"),
    Input("company-dropdown", "value"),
    Input("year-radio", "value"),
)
def pin_selected_report(company, yr):
    dff = df[(df["Company Name"] == company) & (df['Report Year'] == yr)]
    dff = dff.fillna('')
    records = dff.to_dict("records")
    return {"pinnedTopRowData": records}, records


@callback(
    Output("grid", "dashGridOptions", allow_duplicate=True),
    Input("grid", "virtualRowData"),
    prevent_initial_call=True
)
def row_pinning_bottom(data):
    pinned_data = []
    if data:
        dff = pd.DataFrame(data) if data else df
        medians = dff[numeric_columns].median().round(1).to_dict()
        if medians:
            pinned_data = [{"Company": "Median", **medians}]

    grid_option_patch = Patch()
    grid_option_patch["pinnedBottomRowData"] = pinned_data
    return grid_option_patch


@callback(
    Output("grid", "filterModel"),
    Input("year-radio", "value"),
    State("grid", "filterModel"),
)
def update_filter_model(year, model):
    if model:
        model["Report Year"] = {"filterType": "number", "type": "equals", "filter": year}
        return model
    return dash.no_update

@callback(
    Output("bar-chart-card", "children"),
    Input("store-selected", "data")
)
def make_bar_chart(data):
    if data is None or data[0] == {}:
        fig = {}
    else:
        data = data[0]

        # Separate the data for male and female
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        male_percentages = [data[f'{q} Male'] for q in quarters]
        female_percentages = [data[f'{q} Female'] for q in quarters]

        quarter_labels = {
            'Q1': 'Lower (Q1)',
            'Q2': 'Lower Middle (Q2)',
            'Q3': 'Upper Middle (Q3)',
            'Q4': 'Upper (Q4)'
        }
        custom_labels = [quarter_labels[q] for q in quarters]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=custom_labels,
            x=male_percentages,
            name='Male',
            orientation='h',
            marker=dict(color='#19A0AA'),
            text=male_percentages,
            textfont_size=14,
            textposition='inside',  # Position the text inside the bars
        ))

        fig.add_trace(go.Bar(
            y=custom_labels,
            x=female_percentages,
            name='Female',
            orientation='h',
            marker=dict(color='#F15F36'),
            text=female_percentages,
            textfont_size=14,
            textposition='inside',
        ))

        fig.update_layout(
            xaxis=dict(ticksuffix='%'),
            yaxis=dict(title='Quartile', categoryorder='array', categoryarray=quarters),
            barmode='stack',
            template='plotly_white',
            legend=dict(
                orientation='h',  # Horizontal legend
                yanchor='bottom',
                y=-0.25,  # Position below the chart
                xanchor='center',
                x=0.5,  # Centered horizontally
                traceorder='normal'
            ),
            margin = dict(l=10, r=10, t=10, b=10),
        )

    return dbc.Card([
        dbc.CardHeader(html.H2("Proportion of men and women in each pay quartile"), className="text-center"),
        dcc.Graph(figure=fig, style={"height":250}, config={'displayModeBar': False})
    ])


@callback(
    Output("title", "children"),
    Input("store-selected", "data")
)
def make_title(data):
    data=data[0]
    title = f"""
    ## {data["Report Year"]} Gender Pay Gap Report for [{data["Company Name"]}]({data["Company Site"]}) 
    ** For more company-specific details see the report link in the table below **
    """
    return title


@callback(
    Output("paygap-card", "children"),
    Input("store-selected", "data")
)
def make_pay_gap_card(data):
    data=data[0]
    data = {k: (f"{v}%" if v  else '') for k, v in data.items()}
    paygap = dbc.Row([
        dbc.Col([
            html.Div("Hourly Pay Gap", className=" border-bottom border-3"),
            html.Div("ALL"),
            html.Div("Part Time"),
            html.Div("Temporary")
        ], style={"minWidth": 250}),
        dbc.Col([
            html.Div("Mean", className=" border-bottom border-3"),
            html.Div( f"{data['Mean Hourly Gap']}"),
            html.Div(f"{data['Mean Hourly Gap Part Time']}"),
            html.Div(f"{data['Mean Hourly Gap Part Temp']}"),
        ]),
        dbc.Col([
            html.Div("Median", className=" border-bottom border-3"),
            html.Div(f"{data['Median Hourly Gap']}"),
            html.Div(f"{data['Median Hourly Gap Part Time']}"),
            html.Div(f"{data['Median Hourly Gap Part Temp']}"),
        ])
    ], style={"minWidth": 400})

    mean = dbc.Alert(dcc.Markdown(
        f"""
        ** Mean Pay **  
        ### {data['Mean Hourly Gap']}  
        Higher for men
        """,
    ), color="dark")

    median = dbc.Alert(dcc.Markdown(
        f"""
            ** Median Pay ** 
            ### {data['Median Hourly Gap']}  
            Higher for men
            """,
    ), color="dark")

    card =  dbc.Card([
        dbc.CardHeader(html.H2("Hourly Pay Gap"), className="text-center"),
        dbc.CardBody([
            dbc.Row([dbc.Col(mean), dbc.Col(median)], className="text-center"),
            paygap
        ])
    ])
    return card


@callback(
    Output("bonusgap-card", "children"),
    Input("store-selected", "data")
)
def make_bonus_gap_card(data):
    data=data[0]
    if data['Mean Bonus Gap'] == '':
        return ""
    data = {k: (f"{v}%" if v  else '') for k, v in data.items()}
    bonusgap = dbc.Row([
        html.Div("Proportion of employees by gender to receive a bonus:", className="mb-1"),
        dbc.Col([
            html.Div("Bonus and BIK Pay Gap", className=" border-bottom border-3"),
            html.Div("Bonus"),
            html.Div("Benefits In Kind"),

        ], style={"minWidth": 250}),
        dbc.Col([
            html.Div("Men", className=" border-bottom border-3"),
            html.Div( f"{data['Percentage Bonus Paid Male']}"),
            html.Div(f"{data['Percentage BIK Paid Male']}"),

        ]),
        dbc.Col([
            html.Div("Women", className=" border-bottom border-3"),
            html.Div(f"{data['Percentage Bonus Paid Female']}"),
            html.Div(f"{data['Percentage BIK Paid Female']}"),

        ])
    ], style={"minWidth": 400})

    mean = dbc.Alert(dcc.Markdown(
        f"""
        ** Mean Bonus Pay **  
        ### {data['Mean Bonus Gap']}  
        Higher for men
        """,
    ), color="dark")

    median = dbc.Alert(dcc.Markdown(
        f"""
            ** Median Bonus Pay ** 
            ### {data['Median Bonus Gap']}  
            Higher for men
            """,
    ), color="dark")

    card =  dbc.Card([
        dbc.CardHeader(html.H2("Bonus Gap"), className="text-center"),
        dbc.CardBody([
            dbc.Row([dbc.Col(mean), dbc.Col(median)], className="text-center"),
            bonusgap
        ])
    ])
    return card


if __name__ == "__main__":
    app.run(debug=True)