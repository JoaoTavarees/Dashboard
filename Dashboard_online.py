import dash
from sklearn import  metrics
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
from dash import dash_table
from sklearn.feature_selection import SelectKBest, f_regression
from dash.exceptions import PreventUpdate
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np 





power_data = pd.read_csv('Clean_data_Power.csv')
weather_data = pd.read_csv('Weather.csv')
feature_data = pd.read_csv('Features.csv')
data_2019 = pd.read_csv("Data_2019.csv")


# Function to normalize data
def normalize_data(df, columns):
    normalized_df = df.copy()
    for column in columns:
        col_data = df[column]
        min_val = col_data.min()
        max_val = col_data.max()
        normalized_df[column] = (col_data - min_val) / (max_val - min_val)
    return normalized_df

app = dash.Dash(__name__)
server = app.server
app.layout = html.Div(
    children=[
        html.Title(children=["W3.CSS Template"]),
        html.Meta(charSet="UTF-8"),
        html.Meta(name="viewport", content="width=device-width, initial-scale=1"),
        html.Link(rel="stylesheet", href="https://www.w3schools.com/w3css/4/w3.css"),
        html.Link(
            rel="stylesheet", href="https://fonts.googleapis.com/css?family=Poppins"
        ),
        html.Nav(
            className="w3-sidebar w3-red w3-collapse w3-top w3-large w3-padding",
            style={"z-index": "3", "width": "300px", "font-weight": "bold"},
            id="mySidebar",
            children=[
                html.Br(),
                html.A(
                    href="javascript:void(0)",
                    className="w3-button w3-hide-large w3-display-topleft",
                    style={"width": "100%", "font-size": "22px"},
                    children=["Close Menu"],
                ),
                html.Div(
                    className="w3-container",
                    children=[
                        html.H3(
                            className="w3-padding-64",
                            children=[
                                html.B(
                                    children=["João Tavares (100331)"]
                                )
                            ],
                        )
                    ],
                ),
                html.Div(
                    className="w3-bar-block",
                    children=[
                        html.A(
                            href="#introduction",
                            className="w3-bar-item w3-button w3-hover-white",
                            children=["Introduction"],
                        ),
                        html.A(
                            href="#raw_data",
                            className="w3-bar-item w3-button w3-hover-white",
                            children=["Raw Data"],
                        ),
                        html.A(
                            href="#Exploratory_Data",
                            className="w3-bar-item w3-button w3-hover-white",
                            children=["Exploratory Data"],
                        ),
                        html.A(
                            href="#Feature_Selection",
                            className="w3-bar-item w3-button w3-hover-white",
                            children=["Feature Selection"],
                        ),
                        html.A(
                            href="#Regression",
                            className="w3-bar-item w3-button w3-hover-white",
                            children=["Forecast"],
                        ),
                    
                    ],
                ),
            ],
        ),
        html.Header(
            className="w3-container w3-top w3-hide-large w3-red w3-xlarge w3-padding",
            children=[
                html.A(
                    href="javascript:void(0)",
                    className="w3-button w3-red w3-margin-right",
                    children=["☰"],
                ),
                html.Span(children=["Welcome"]),
            ],
        ),
        html.Div(
            className="w3-overlay w3-hide-large",
            style={"cursor": "pointer"},
            title="close side menu",
            id="myOverlay",
        ),
        html.Div(
            className="w3-main",
            style={"margin-left": "340px", "margin-right": "40px"},
            children=[
                html.Div(
                    className="w3-container",
                    style={"margin-top": "80px"},
                    id="showcase",
                    children=[
                        html.H1(
                            className="w3-jumbo",
                            children=[html.B(children=["Energy Services Forecast"])],
                        )
                    ],
                ),
                html.Div(
                    className="w3-container",
                    id="introduction",
                    style={"margin-top": "75px"},
                    children=[
                        html.H1(
                            className="w3-xxxlarge w3-text-red",
                            children=[html.B(children=["Introduction"])],
                        ),
                        html.Hr(
                            style={"width": "50px", "border": "5px solid red"},
                            className="w3-round",
                        ),
                        html.P(
                            children=[
                                "This dashboard aims to represent the data that the user wishes to display in a more aesthetically pleasing structure as well as forecast the 2019 Power consumption values of the Civil Building"
                            ]
                        ),
                        html.P(
    children=[
        "It is divided into 4 sections:",
        html.Ul([
            html.Li("In the Raw Data section, users can visualize graphical displays of Power and weather conditions (choose the variables you wish) conditions in 2017 and 2018."),
            html.Li("In the Exploratory Data section, users can select different variables (such as weather) and visualize them, in two different types of plots with a statistics table under the graphical display."),
            html.Li("In the Feature Selection section, users can choose the variables to perform the feature selection from various methods."),
            html.Li("In the Forecast section, users can select the variables to perform the forecast of the 2019 Power consumption. 3 different regression models are available to select as well as the metrics the user wishes to calculate."),
        ])
    ]
)

                    ],
                ),

                # "Raw Data" div
                html.Div(
                    className="w3-container",
                    id="raw_data",
                    children=[
                        html.H1(
                            className="w3-xxxlarge w3-text-red",
                            children=[html.B(children=["Raw Data"])],
                        ),
                        html.Hr(
                            style={"width": "50px", "border": "5px solid red"},
                            className="w3-round",
                        ),
                        
                        # Power consumption plot inside the "Raw Data" div
                        html.Div(
                            children=[
                                html.H2(
                                    className="w3-text-red",
                                    children=["Power Consumption"],
                                ),
      
                                dcc.Graph(
                                    id="power_consumption_graph",
                                    figure=px.line(power_data, x='Date', y='Power_kW')
                                )
                            ],
                        ),
                    
                  

                # Temperature trend plot inside the "Raw Data" div
                html.Div(
                    children=[
                        html.H2(
                            className="w3-text-red",
                            children=["Weather Variables"],
                        ),

                        dcc.Dropdown(
                            id='weather_variable_dropdown',
                            options=[
                                {'label': 'Temperature', 'value': 'Temperature (C)'},
                                {'label': 'Humidity', 'value': 'Humidity (%)'},
                                {'label': 'Rain', 'value': 'rain (mm/h)'},
                                {'label': 'Pressure', 'value': 'Pressure (mbar)'},
                                {'label': 'Wind Speed', 'value': 'WindSpeed (m/s)'},
                                {'label': 'Solar Radiance', 'value': 'SolarRad (W/m2)'},
                            ],
                            value='Temperature (C)',  # Default value
                            clearable= False,  # Display labels in block style
                            style={'width': '150px', 'backgroundColor': '#c4e3ff', 'color': '#000000', 'borderRadius': '8px'}

                        ),
                         dcc.Graph(
                            id="weather_variable_graph",
                        )
                    ],
                ),
            ]
        ),
    

            
                # Corrected layout code
# Corrected layout code for Select dropdown
html.Div(
    className="w3-container",
    id="Exploratory_Data",
    style={"margin-top": "75px"},
    children=[
        html.H1(
            className="w3-xxxlarge w3-text-red",
            children=[html.B(children=["Exploratory Data"])],
        ),
        html.Hr(
            style={"width": "50px", "border": "5px solid red"},
            className="w3-round",
        ),
        html.P(
            children=[
                "In this section you are able to explore the data and take conclusions of it. You can make two different types of graphs (boxplot and linegraph)"
                " You can also choose which variables you want to visualize and their statistics in a table below the graphic. Don't forget to normalize the data when needed. "
                
            ]
        ),
        html.Div(
    id="plot_type_select",
    children=[
        html.Label("Select Plot Type:"),
        dcc.RadioItems(
            id="plot_type_radio",
            options=[
                {"label": "Line Graph", "value": "line"},
                {"label": "Box Plot", "value": "box"}
            ],
            value="line",  # Default to line graph
            labelStyle={"margin-right": "15px"}
        )
    ]
),
        dcc.Dropdown(
            id="variable_select",
            options=[
                                {'label': 'Temperature', 'value': 'Temperature (C)'},
                                {'label': 'Humidity', 'value': 'Humidity (%)'},
                                {'label': 'Rain', 'value': 'rain (mm/h)'},
                                {'label': 'Pressure', 'value': 'Pressure (mbar)'},
                                {'label': 'Wind Speed', 'value': 'WindSpeed (m/s)'},
                                {'label': 'Solar Radiance', 'value': 'SolarRad (W/m2)'},
            ],
            multi=True,
            value=["Temperature (C)"],  # Default value
        ),
        dcc.Graph(id="exploratory_plot"),
        dcc.Checklist(
                    id="normalize_checkbox",
                    options=[
                        {"label": "Normalize Data", "value": "normalize"}
                    ],
                    value=[],
                    labelStyle={"display": "inline-block", "padding-left": "20px"},
                ),
        # Add DataTable component to layout
# Graph for displaying exploratory plot
        # DataTable component for displaying variable statistics
        html.Div(
            [
                dash_table.DataTable(
                    id="variable_stats_table",
                    columns=[
                        {"name": "Variable", "id": "Variable", "type": "text"},
                        {"name": "Statistic", "id": "Statistic", "type": "text"},
                        {"name": "Value", "id": "Value", "type": "numeric", "format": {"specifier": ".2f"}},
                    ],
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_cell={
                        'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',
                        'whiteSpace': 'normal',
                        'textAlign': 'center'
                    },
                    page_size=10,
                    sort_action="native",
                    sort_mode="multi",
                    filter_action="native",
                    page_action="native"
                )
            ]
        )
    ],
),







                html.Div(
            className="w3-container",
            id="Feature_Selection",
            style={"margin-top": "75px"},
            children=[
                html.H1(
                    className="w3-xxxlarge w3-text-red",
                    children=[html.B(children=["Feature Selection"])],
                ),
                html.Hr(
                    style={"width": "50px", "border": "5px solid red"},
                    className="w3-round",
                ),
                html.P(
                    children=["In this section, you may choose which variables you wish to perform the feature selection on and also choose from the available methods.",
                        html.Div("Performing feature selection using Random Forest Regressor may take a while...",
                                    style={'font-style': 'italic', 'margin-bottom': '10px'}),
                    ]
                ),
                # Dropdown to select variables
                html.Div([
                    html.Label('Variables Selection:'),
    dcc.Dropdown(
        id='variable-dropdown',
        options=[{'label': col, 'value': col} for col in feature_data.columns if col not in ["Date", "Power (kW)"]],
        multi=True,
        value=[],  # Initially no variables selected
        placeholder="Select variables for feature selection",
    ),
                ]),
    html.Div([ html.Label('Feature Selection Method:'),           
    dcc.Dropdown(
        id='feature-method-dropdown',
        options=[
            {'label': 'kBest', 'value': 'kBest'},
            {'label': 'RFE', 'value': 'RFE'},
            {'label': 'Random Forest Regressor', 'value': 'Random Forest Regressor'}
        ],
        value='kBest',  # Default value
        clearable=False
    ),
    ]),
    html.Div([html.Label('                 '),]),
    html.Div(style={'display': 'flex', 'align-items': 'center'},
    children=[
        html.Label('Enter k value:', id='k-value-label', style={'margin-right': '10px'}),
        dcc.Input(id='k-value-input', type='number', value=4, style={'width': '10px'}),
    ]
),
html.Div([html.Label('                 '),]),
html.Div([
    html.Button('Perform Feature Selection', id='feature-selection-button', n_clicks=0),
    html.Div([html.Label('                 '),  ]),
    html.Div( id='feature-selection-result',children=[]),
    ])
            ]
        ),
                
     
           



html.Div(
            className="w3-container",
            id="Regression",
            style={"margin-top": "75px"},
            children=[
                html.H1(
                    className="w3-xxxlarge w3-text-red",
                    children=[html.B(children=["Forecast"])],
                ),
                html.Hr(
                    style={"width": "50px", "border": "5px solid red"},
                    className="w3-round",
                ),
                 html.P(
                    children=["In this section, you may choose which variables you wish to use for the forecast, select the model, and decide on the metrics you wish to calculate. If your forecast is accurate enough, you will receive a congratulations message. Give it a try! :)",
                        html.Div("Performing forecast using Random Forest Regression may take a while...",
                                    style={'font-style': 'italic', 'margin-bottom': '10px'}),
                    ]
                ),
                # Dropdown to select variables
                html.Div([
                    html.Label('Variables Selection:'),
    dcc.Dropdown(
        id='forecast-variable-dropdown',
        options=[{'label': col, 'value': col} for col in feature_data.columns if col not in ["Date", "Power (kW)"]],
        multi=True,
        value=["Power-1 (kW)"], 
        placeholder="Select variables for forecast",
    ),
                ]),
    html.Div([ html.Label('Feature Selection Method:'),           
    dcc.Dropdown(
            id='forecast-method-dropdown',
            options=[
                {'label': 'Linear Regression', 'value': 'linear_regression'},
                {'label': 'Random Forest Regression', 'value': 'random_forest'},
                {'label': 'Decision Tree Regression', 'value': 'decision_tree'},
                # Add options for other regression methods if needed
            ],
            value='linear_regression',
            placeholder="Select regression method",
        ),
    ]),
    dcc.Dropdown(
            id='metrics-dropdown',
            multi=True,
            options=[
                {'label': 'Mean Absolute Error (MAE)', 'value': 'mae'},
                {'label': 'Mean Squared Error (MSE)', 'value': 'mse'},
                {'label': 'Mean Bias Error (MBE)', 'value': 'mbe'},
                {'label': 'Root Mean Squared Error (RMSE)', 'value': 'rmse'},
                {'label': 'cv Root Mean Squared Error (cvRMSE)', 'value': 'cvrmse'},
                {'label': 'Normalized Mean Bias Error (NMBE)', 'value': 'nmbe'},
                # Add options for other evaluation metrics if needed
            ],
            value='mse',
            placeholder="Select evaluation metric",
        ),
        html.Button('Perform Forecast', id='forecast-button', n_clicks=0),
    ]),
    
    dcc.Graph(id ="forecast-graph"),

    # Forecast result section
    html.Div(id='forecast-result'),
html.Div(id='congratulatory-message', style={'color': 'green', 'font-size': '18px'}),



    
                
     ],
           ),
    ]
)

@app.callback(
    Output('weather_variable_graph', 'figure'),
    [Input('weather_variable_dropdown', 'value')]
)
def update_weather_variable_graph(selected_variable):
    if selected_variable:
        fig = px.line(weather_data, x='Date', y=selected_variable)
        return fig
    else:
        return {}

# Callback to update the exploratory plot based on selected variables
# Add a callback to update the graph based on the selected plot type and variable
import plotly.graph_objs as go

@app.callback(
    Output("exploratory_plot", "figure"),
    [
        Input("plot_type_radio", "value"),
        Input("variable_select", "value"),
        Input("normalize_checkbox", "value"),
    ],
)
def update_exploratory_plot(plot_type, selected_variable, normalize_option):
    if not selected_variable:
        return {
            "data": [],
            "layout": {
                "title": "Exploratory Data",
                "annotations": [
                    {
                        "text": "Please select at least one variable.",
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.5,
                        "y": 0.5,
                        "showarrow": False,
                        "font": {"size": 20, "color": "red"},
                    }
                ],
            },
        }

    # Copy the original data to avoid modifying the original DataFrame
    data_copy = weather_data[selected_variable].copy()

    if plot_type == "line":
        # Generate line plot
        fig = px.line(weather_data, x="Date", y=selected_variable, color_discrete_sequence=px.colors.qualitative.Plotly)
        # Update x-axis label for line plot
        fig.update_xaxes(title="Date", tickfont=dict(size=12, color='black'), showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
        # Update y-axis label for line plot
        fig.update_yaxes(title="Value", tickfont=dict(size=12, color='black'), showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
    elif plot_type == "box":
        # Generate box plot
        if isinstance(selected_variable, str):
            selected_variable = [selected_variable]
        fig = px.box(weather_data[selected_variable], y=selected_variable, title="Boxplot", color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_xaxes(title="Variable", tickfont=dict(size=12, color='black'), showgrid=True, gridwidth=0.5, gridcolor='lightgrey')

        # Update y-axis label for box plot
        fig.update_yaxes(title="Value", tickfont=dict(size=12, color='black'), showgrid=True, gridwidth=0.5, gridcolor='lightgrey')

  
    
    # Check if the normalization option is selected and the plot type is not "line"
    # Normalize the data if the checkbox is checked
    if "normalize" in normalize_option:
        weather_data_normalized = normalize_data(weather_data[selected_variable], selected_variable)
        if plot_type == "line":
            fig = px.line(weather_data_normalized, x=weather_data["Date"], y=selected_variable, color_discrete_sequence=px.colors.qualitative.Plotly)
            fig.update_xaxes(title="Date", tickfont=dict(size=12, color='black'), showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
            fig.update_yaxes(title="Normalized Value", tickfont=dict(size=12, color='black'), showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
            fig.update_layout(title="Exploratory Data")
        if plot_type == "box":
            fig = px.box(weather_data_normalized[selected_variable], y=selected_variable, title="Boxplot", color_discrete_sequence=px.colors.qualitative.Plotly)
            fig.update_layout(title="Exploratory Data" , yaxis_title="Normalized Value")
            fig.update_yaxes(title="Normalized Value", tickfont=dict(size=12, color='black'), showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
            fig.update_xaxes(title="Variable", tickfont=dict(size=12, color='black'), showgrid=True, gridwidth=0.5, gridcolor='lightgrey')


 # Update the layout of the plot
    fig.update_layout(
        title="Exploratory Data",
        legend_title="Variables",
        legend=dict(title_font=dict(size=14)),
        font=dict(family="Arial, sans-serif", size=12, color="black"),
        plot_bgcolor="white",  # Background color
        margin=dict(l=50, r=50, t=80, b=50),  # Margins
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
    )

    return fig



# Callback to update the table
@app.callback(
    Output("variable_stats_table", "data"),
    [Input("variable_select", "value")]
)
def update_variable_stats_table(selected_variable):
    if not selected_variable:
        return []
    
    # Calculate statistics for selected variables
    stats_data = []
    for var in selected_variable:
        var_stats = weather_data[var].describe().reset_index()
        var_stats.rename(columns={"index": "Statistic", var: "Value"}, inplace=True)
        var_stats["Variable"] = var
        stats_data.append(var_stats.to_dict("records"))
    
    return [item for sublist in stats_data for item in sublist]



@app.callback(
    Output('feature-method-specific-options', 'children'),
    [Input('feature-method-dropdown', 'value')]
)
def render_specific_options(method):
    if method == 'kBest':
        return dcc.Input(id='k-value-input', type='number', value=4)
    elif method == 'RFE':
        return dcc.Input(id='k-value-input', type='number', value=4)
    elif method == 'Random Forest Regressor':
        return dcc.Input(id='k-value-input', type='number', value=4)
    

@app.callback(
    Output('feature-selection-result', 'children'),
    [Input('feature-selection-button', 'n_clicks')],
    [State('feature-method-dropdown', 'value'),
     State('variable-dropdown', 'value'),
     State('k-value-input', 'value')]
)
def update_feature_selection_result(n_clicks, method, selected_variables, k_value):
    if n_clicks == 0:
        return dash.no_update
    
    if method == 'kBest':
        if not selected_variables:
            return "Please select at least one variable."
        
        # Perform feature selection using SelectKBest
        # Perform feature selection using SelectKBest
        X = feature_data[selected_variables].values  # Use only selected variables
        Y = feature_data['Power (kW)'].values  # Assuming 'Power' is your target column
        features = SelectKBest(k=k_value, score_func=f_regression)
        fit = features.fit(X, Y)
        scores = fit.scores_

        # Assigning different colors to each variable
        colors = px.colors.qualitative.Plotly[:len(selected_variables)]

        # Plot the histogram of selected features
        fig = go.Figure()
        for variable, score, color in zip(selected_variables, scores, colors):
            fig.add_trace(go.Bar(x=[variable], y=[score], showlegend=False))

        fig.update_layout(title='Selected Features Histogram', xaxis_title='Feature', yaxis_title='Score')

     # Plot the histogram of selected features
       
        return dcc.Graph(figure=fig)
    
    elif method == 'RFE':
        if not selected_variables:
            return "Please select at least one variable."
        # Perform feature selection using RFE
        model = LinearRegression()
        rfe = RFE(model, n_features_to_select=k_value)
        X = feature_data[selected_variables]  # Use only selected variables
        Y = feature_data['Power (kW)']  # Assuming 'Power' is your target column
        fit = rfe.fit(X, Y)
        
        # Get the ranking of features
        rankings = fit.ranking_
        
        # Select top k features based on their rankings
        selected_features = [selected_variables[i] for i in range(len(selected_variables)) if rankings[i] == 1]
        
        # Return a string listing the selected features
        selected_features_str = ", ".join(selected_features)
        return f"The top {k_value} features selected by RFE are: {selected_features_str}"
    
    elif method == 'Random Forest Regressor':
        if not selected_variables:
            return "Please select at least one variable."
        
        # Perform feature selection using RandomForestRegressor
        X = feature_data[selected_variables]  # Use only selected variables
        Y = feature_data['Power (kW)']  # Assuming 'Power' is your target column

    # Create a RandomForestRegressor model
        model = RandomForestRegressor()
        model.fit(X, Y)
        feature_importances = model.feature_importances_

    # Create a DataFrame to store the feature importances along with their variable names
        feature_importances_df = pd.DataFrame({'Feature': selected_variables, 'Importance': feature_importances})

    # Sort the DataFrame by importance values in descending order
        feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

    # Plot the histogram of feature importances
        fig = px.bar(feature_importances_df, x='Feature', y='Importance', title='Feature Importances (Random Forest)',
                 labels={'Feature': 'Feature', 'Importance': 'Importance Score'})

        fig.update_layout(xaxis_title='Feature', yaxis_title='Importance Score')

    # Return the histogram plot
        return dcc.Graph(id='feature-importance-histogram', figure=fig)


@app.callback(
    Output('k-value-input', 'style'),
    [Input('feature-method-dropdown', 'value')]
)
def hide_k_value_input(method):
    if method == 'RFE':
        return {'display': 'block'}  # Hide the input for k-value

    else:
        return {'display': 'none'}  # Show the input for other methods



@app.callback(
    Output('k-value-label', 'style'),
    [Input('feature-method-dropdown', 'value')]
)
def update_k_value_label_visibility(method):
    if method == 'RFE':
        return {'display': 'inline-block', 'margin-right': '10px'}
    else:
        return {'display': 'none'}
    


metric_results = []

@app.callback(
    [Output('forecast-result', 'children'),
     Output('forecast-graph', 'figure'),
     Output('congratulatory-message', 'children')],  
    [Input('forecast-button', 'n_clicks')],
    [State('forecast-method-dropdown', 'value'),
     State('forecast-variable-dropdown', 'value'),
     State('metrics-dropdown', 'value')]
)
def perform_forecast(n_clicks, method, selected_variables, metric):

    if method == 'linear_regression':
        # Perform linear regression forecast
        y_pred = perform_linear_regression_forecast(selected_variables)
        forecast_title = 'Linear Regression Forecast'
    elif method == 'random_forest':
        # Perform Random Forest forecast
        y_pred = perform_random_forest_forecast(selected_variables)
        forecast_title = 'Random Forest Forecast'
    elif method == "decision_tree":
        y_pred = perform_decision_tree_forecast(selected_variables)
        forecast_title = "Decision Tree Forecast"
    else:
        return "Method not implemented yet"

    # Calculate selected metrics
    metric_results = []

    if 'mae' in metric:
            value = metrics.mean_absolute_error(data_2019['Power (kW)'], y_pred)
            metric_results.append({'Metric': 'Mean Absolute Error (MAE)', 'Value': value})
    if 'mse' in metric:
            value = metrics.mean_squared_error(data_2019['Power (kW)'], y_pred)
            metric_results.append({'Metric': 'Mean Squared Error (MSE)', 'Value': value})
    if 'mbe' in metric:
            value = np.mean(data_2019['Power (kW)'] - y_pred)
            metric_results.append({'Metric': 'Mean Bias Error (MBE)', 'Value': value})
    if  'rmse' in metric:
            value = np.sqrt(metrics.mean_squared_error(data_2019['Power (kW)'], y_pred))
            metric_results.append({'Metric': 'Root Mean Squared Error (RMSE)', 'Value': value})
    if  'cvrmse' in metric:
            value = np.sqrt(metrics.mean_squared_error(data_2019['Power (kW)'], y_pred))/ np.mean(data_2019['Power (kW)'])
            metric_results.append({'Metric': 'cv Root Mean Squared Error (cvRMSE)', 'Value': value})
    if 'nmbe' in metric:
            value = np.mean(data_2019['Power (kW)'] - y_pred)/ np.mean(data_2019['Power (kW)'])
            metric_results.append({'Metric': 'Normalized Mean Bias Error (NMBE)', 'Value': value})
    
    # Check if cvRMSE and NMVE meet the conditions
    cvrmse = next((result['Value'] for result in metric_results if result['Metric'] == 'cv Root Mean Squared Error (cvRMSE)'), None)
    nmbe = next((result['Value'] for result in metric_results if result['Metric'] == 'Normalized Mean Bias Error (NMBE)'), None)
    
    message = None
    if cvrmse is not None and nmbe is not None:
        if  cvrmse <= 0.3 and nmbe <= 0.1:
            message = "Congratulations! Your forecast respects the best of the benchmarks!"






    # Create table with metric results
    table_style = {
    'border': '1px solid #ddd',
    'border-collapse': 'collapse',
    'width': '100%',
    'margin-top': '10px'
}

    header_style = {
    'background-color': '#f2f2f2',
    'border': '1px solid #ddd',
    'padding': '8px',
    'text-align': 'left'
}

    cell_style = {
    'border': '1px solid #ddd',
    'padding': '8px',
    'text-align': 'left'
}

    table = html.Table(
    style=table_style,
    children=[
        html.Thead(
            html.Tr(
                [html.Th(col, style=header_style) for col in ['Metric', 'Value']]
            )
        ),
        html.Tbody(
            [html.Tr([html.Td(result['Metric'], style=cell_style), html.Td(result['Value'], style=cell_style)]) for result in metric_results]
        )
    ]
)
    # Plot the forecast results along with the real values
    real_values_trace = go.Scatter(x=data_2019["Date"], y=data_2019['Power (kW)'], mode='lines', name='Real Values', line=dict(color='orange'))
    forecast_trace = go.Scatter(x=data_2019["Date"], y=y_pred, mode='lines', name='Forecast', line=dict(color='blue'))

    layout = go.Layout(
        title='Power Consumption Forecast for 2019 using ' + forecast_title,
        xaxis=dict(title='Date'),
        yaxis=dict(title='Power Consumption (kW)')
    )

    graph_figure = {
        'data': [real_values_trace, forecast_trace],
        'layout': layout
    }






    return table, graph_figure, message

def perform_linear_regression_forecast(selected_variables):
    
        # Split the power_data into training (2017 and 2018) and testing (2019) sets
        feature_data['Date'] = pd.to_datetime(feature_data['Date']) 
        data_2019['Date'] = pd.to_datetime(data_2019['Date']) 

        train_data = feature_data[(feature_data['Date'].dt.year == 2017) | (feature_data['Date'].dt.year == 2018)]
        test_data = data_2019
  
        # Split the training and testing data into features (X) and target (y)
        X_train = train_data[selected_variables]
        y_train = train_data['Power (kW)']
        X_test = test_data[selected_variables]

        # Train the regression model
        regr = LinearRegression()
        regr.fit(X_train, y_train)

        return regr.predict(X_test)


def perform_random_forest_forecast(selected_variables):
    feature_data['Date'] = pd.to_datetime(feature_data['Date']) 
    data_2019['Date'] = pd.to_datetime(data_2019['Date']) 
    # Split the power_data into training (2017 and 2018) and testing (2019) sets
    train_data = feature_data[(feature_data['Date'].dt.year == 2017) | (feature_data['Date'].dt.year == 2018)]
    test_data = data_2019
    

    # Split the training and testing data into features (X) and target (y)
    X_train = train_data[selected_variables]
    y_train = train_data['Power (kW)']
    X_test = test_data[selected_variables]

    # Define Random Forest parameters
    parameters = {
        'bootstrap': True,
        'min_samples_leaf': 3,
        'n_estimators': 200,
        'min_samples_split': 15,
        'max_features': 'sqrt',
        'max_depth': 20,
        'max_leaf_nodes': None
    }

    # Create and train the Random Forest model
    RF_model = RandomForestRegressor(**parameters)
    RF_model.fit(X_train, y_train)

    # Make predictions for 2019 data
    return RF_model.predict(X_test)

def perform_decision_tree_forecast(selected_variables):
    from sklearn.tree import DecisionTreeRegressor
    feature_data['Date'] = pd.to_datetime(feature_data['Date']) 
    data_2019['Date'] = pd.to_datetime(data_2019['Date']) 
    # Split the power_data into training (2017 and 2018) and testing (2019) sets
    train_data = feature_data[(feature_data['Date'].dt.year == 2017) | (feature_data['Date'].dt.year == 2018)]
    test_data = data_2019
    

    # Split the training and testing data into features (X) and target (y)
    X_train = train_data[selected_variables]
    y_train = train_data['Power (kW)']
    X_test = test_data[selected_variables]

    DT_regr_model = DecisionTreeRegressor()

    DT_regr_model.fit(X_train, y_train)

    return DT_regr_model.predict(X_test)



if __name__ == "__main__":
    app.run_server()
