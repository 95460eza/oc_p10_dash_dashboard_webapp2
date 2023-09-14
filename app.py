
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_bootstrap_components as dbc  # To define rows and columns on the page
import dash_mantine_components as dmc  # To define a grid on the page within which to insert dmc.Cols and define their width by assigning a number to the span property.
import pandas as pd
import plotly.express as px
import os


# Creates the DASH INTERACTIVE web app OBJECT (content is interactive and will be seen in an browser)
# Incorporate styling Dash Bootstrap Components
# external_stylesheets = [dbc.themes.CERULEAN]
# OR
# Incorporate styling Dash Mantine Components
external_stylesheets = [dmc.theme.DEFAULT_COLORS]
dash_app = Dash(__name__, external_stylesheets=external_stylesheets)

# Specific to Azure:  Azure is ALSO looking for a Flask app variable named app!!!
app = dash_app.server


# Read in Data
file_to_load = os.path.join( os.getcwd(), "dash_texts_data/df_airline_tweets.csv")
df = pd.read_csv(file_to_load)[ ['airline', 'airline_sentiment_confidence', 'negativereason_confidence', 'retweet_count']]


# SETUP CALLBACK: It Enables the use of control components for building the interaction
@callback(
    # You must provide all Outputs first. "component_property" are the parameters whose values the interaction will change
    Output(component_id='graph-to-display', component_property='figure'),
    Input(component_id='radio-button-to-select-column', component_property='value'),

)
def update_graph(col_chosen):
    fig = px.histogram(df, x='airline', y=col_chosen, histfunc='avg')
    return fig


# Define the Web App Layout
dash_app.layout = dbc.Container([  dmc.Title('Welcome DATA Apprentice', color="blue", size="h3"),
                                   dmc.RadioGroup( [dmc.Radio(i, value=i) for i in ['airline_sentiment_confidence','negativereason_confidence', 'retweet_count']],
                                                   id='radio-button-to-select-column',
                                                   value='retweet_count', size="sm"
                                                 ),
                                   dmc.Grid([  dmc.Col( [ dash_table.DataTable( data=df.to_dict('records'), page_size=12, style_table={'overflowX': 'auto'} ) ],
                                                        span=6
                                                        ),
                                 dmc.Col([dcc.Graph(figure={}, id='graph-to-display')], span=6)

                             ]

                             ),  # "," not needed

                             ], fluid=True

                            )




if __name__ == '__main__':
    dash_app.run_server(debug=True)
#    app.run(debug=True)
#   app.run(host='0.0.0.0', port=5000, debug=True)
