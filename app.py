

#import warnings
#import os
#import io
#import random
#import base64
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_bootstrap_components as dbc  # To define rows and columns on the page
import dash_mantine_components as dmc  # To define a grid on the page within which to insert dmc.Cols and define their width by assigning a number to the span property.
import pandas as pd
#import numpy as np
#import plotly.express as px
#import matplotlib.pyplot as plt
#import nltk
# Download "stop words" list!
#nltk.download("stopwords")
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
# Download punctuation remover!
#nltk.download("punkt")
#import re
#from wordcloud import WordCloud


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
#file_to_load = os.path.join( os.getcwd(), "dash_texts_data/df_airline_tweets.csv")
#df = pd.read_csv(file_to_load)
# Convert the colunm from a pandas SERIES object to a CATEGORY object
#df["airline_sentiment"] = pd.Categorical(df["airline_sentiment"])
# Create a new NUMERIC column by converting category values to integers
#df["sentiment"] = df["airline_sentiment"].cat.codes
#df["tweet_length"] = df["text"].apply(lambda x: len(x))
#df = df.sort_values(by=["tweet_created"])




if __name__ == '__main__':
    dash_app.run_server(debug=True)
#   app.run(debug=True)
#   app.run(host='0.0.0.0', port=5000, debug=True)
