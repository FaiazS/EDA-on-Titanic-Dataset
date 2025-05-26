import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Load the Titanic dataset from Plotly
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/titanic.csv')

# Clean up column names to match the original code
df = df.rename(columns={
    'Pclass': 'class',
    'Sex': 'sex',
    'Age': 'age',
    'Survived': 'survived',
    'Fare': 'fare',
    'Name': 'name',
    'Embarked': 'embark_town',
    'Cabin': 'cabin',
    'Ticket': 'ticket',
    'SibSp': 'sibsp',
    'Parch': 'parch',
    'PassengerId': 'passenger_id'
})

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Titanic Dataset Explorer", className="text-center my-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Select Features:", className="mb-3"),
            dcc.Dropdown(
                id='x-feature',
                options=[{'label': col, 'value': col} for col in ['age', 'fare', 'class', 'sex', 'survived', 'embark_town']],
                value='age',
                className="mb-3"
            ),
            dcc.Dropdown(
                id='y-feature',
                options=[{'label': col, 'value': col} for col in ['age', 'fare', 'class', 'sex', 'survived', 'embark_town']],
                value='fare',
                className="mb-3"
            ),
            dcc.Dropdown(
                id='color-feature',
                options=[{'label': col, 'value': col} for col in ['sex', 'class', 'survived', 'embark_town', None]],
                value='sex',
                className="mb-3"
            ),
            dcc.Dropdown(
                id='chart-type',
                options=[
                    {'label': 'Scatter Plot', 'value': 'scatter'},
                    {'label': 'Box Plot', 'value': 'box'},
                    {'label': 'Violin Plot', 'value': 'violin'},
                    {'label': 'Histogram', 'value': 'histogram'}
                ],
                value='scatter',
                className="mb-3"
            ),
            html.Hr(),
            html.H4("Filters:", className="mb-3"),
            dcc.RangeSlider(
                id='age-slider',
                min=0,
                max=80,
                step=1,
                value=[0, 80],
                marks={0: '0', 20: '20', 40: '40', 60: '60', 80: '80'},
                className="mb-3"
            ),
            html.Div(id='age-slider-output', className="mb-3"),
            dcc.Checklist(
                id='class-filter',
                options=[{'label': c, 'value': c} for c in df['class'].unique()],
                value=df['class'].unique().tolist(),
                className="mb-3"
            ),
        ], width=3, className="bg-light p-3"),
        
        dbc.Col([
            dcc.Graph(id='main-graph', className="mb-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='pie-chart'), width=6),
                dbc.Col(dcc.Graph(id='survival-rate'), width=6)
            ])
        ], width=9)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Data Table"),
            html.Div(id='data-table', className="table-responsive")
        ], width=12)
    ])
], fluid=True)

# Callbacks
@callback(
    Output('age-slider-output', 'children'),
    Input('age-slider', 'value')
)
def update_age_output(value):
    return f'Age Range: {value[0]} - {value[1]} years'

@callback(
    Output('main-graph', 'figure'),
    [Input('x-feature', 'value'),
     Input('y-feature', 'value'),
     Input('color-feature', 'value'),
     Input('chart-type', 'value'),
     Input('age-slider', 'value'),
     Input('class-filter', 'value')]
)
def update_graph(x_feature, y_feature, color_feature, chart_type, age_range, class_filter):
    filtered_df = df[(df['age'] >= age_range[0]) & 
                    (df['age'] <= age_range[1]) &
                    (df['class'].isin(class_filter))]
    
    if chart_type == 'scatter':
        fig = px.scatter(
            filtered_df, 
            x=x_feature, 
            y=y_feature, 
            color=color_feature,
            title=f'{y_feature.capitalize()} vs {x_feature.capitalize()}',
            hover_data=['name', 'age', 'fare', 'class', 'sex', 'survived']
        )
    elif chart_type == 'box':
        fig = px.box(
            filtered_df, 
            x=x_feature, 
            y=y_feature, 
            color=color_feature,
            title=f'Distribution of {y_feature.capitalize()} by {x_feature.capitalize()}'
        )
    elif chart_type == 'violin':
        fig = px.violin(
            filtered_df, 
            x=x_feature, 
            y=y_feature, 
            color=color_feature,
            box=True,
            title=f'Distribution of {y_feature.capitalize()} by {x_feature.capitalize()}'
        )
    else:  # histogram
        fig = px.histogram(
            filtered_df, 
            x=x_feature, 
            color=color_feature,
            marginal='box',
            title=f'Distribution of {x_feature.capitalize()}',
            barmode='overlay'
        )
    
    fig.update_layout(transition_duration=500)
    return fig

@callback(
    Output('pie-chart', 'figure'),
    [Input('age-slider', 'value'),
     Input('class-filter', 'value')]
)
def update_pie_chart(age_range, class_filter):
    filtered_df = df[(df['age'] >= age_range[0]) & 
                    (df['age'] <= age_range[1]) &
                    (df['class'].isin(class_filter))]
    
    survival_count = filtered_df['survived'].value_counts().reset_index()
    survival_count['survived'] = survival_count['survived'].map({0: 'Did Not Survive', 1: 'Survived'})
    
    fig = px.pie(
        survival_count, 
        values='count', 
        names='survived',
        title='Survival Distribution',
        hole=0.4
    )
    return fig

@callback(
    Output('survival-rate', 'figure'),
    [Input('age-slider', 'value'),
     Input('class-filter', 'value')]
)
def update_survival_rate(age_range, class_filter):
    filtered_df = df[(df['age'] >= age_range[0]) & 
                    (df['age'] <= age_range[1]) &
                    (df['class'].isin(class_filter))]
    
    survival_rate = filtered_df.groupby('class')['survived'].mean().reset_index()
    
    fig = px.bar(
        survival_rate, 
        x='class', 
        y='survived',
        title='Survival Rate by Class',
        labels={'survived': 'Survival Rate', 'class': 'Passenger Class'}
    )
    fig.update_yaxes(tickformat=".0%")
    return fig

@callback(
    Output('data-table', 'children'),
    [Input('age-slider', 'value'),
     Input('class-filter', 'value')]
)
def update_table(age_range, class_filter):
    filtered_df = df[(df['age'] >= age_range[0]) & 
                    (df['age'] <= age_range[1]) &
                    (df['class'].isin(class_filter))]
    
    return dbc.Table.from_dataframe(
        filtered_df[['name', 'age', 'sex', 'class', 'fare', 'survived']].head(10),
        striped=True, 
        bordered=True, 
        hover=True,
        responsive=True
    )

if __name__ == '__main__':
    app.run(debug=True)
