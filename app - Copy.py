

import warnings
import os
import io
import random
import base64
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_bootstrap_components as dbc  # To define rows and columns on the page
import dash_mantine_components as dmc  # To define a grid on the page within which to insert dmc.Cols and define their width by assigning a number to the span property.
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Download punctuation remover!
nltk.download("punkt")
import re
from wordcloud import WordCloud

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
df = pd.read_csv(file_to_load)
# Convert the colunm from a pandas SERIES object to a CATEGORY object
df["airline_sentiment"] = pd.Categorical(df["airline_sentiment"])
# Create a new NUMERIC column by converting category values to integers
df["sentiment"] = df["airline_sentiment"].cat.codes
df = df.sort_values(by=["tweet_created"])
#columns_to_keep = ['airline', 'retweet_count', 'sentiment']

# Sentiment Category Proportion by Airline
grouped = (
    df.groupby(["airline", "airline_sentiment"])[
        "airline_sentiment"
    ].count()
    / df.groupby(["airline"])["airline_sentiment"].count()
)

grouped = pd.DataFrame(grouped)
grouped.columns = ["proportion"]
grouped = grouped.reset_index()

df_tokenize = df.copy()

df_tokenize["tokenized_tweet"] = df["text"].apply(lambda x: word_tokenize(re.sub(r"[^\w\s]", "", x)))
df_tokenize = df_tokenize[["tweet_id", "airline", "tokenized_tweet"]]

# Make strings for Words Cloud
def make_string_for_wordcloud(input_dict):
    
    stop_words = set(stopwords.words("english"))
    concat_tokens = []

    for token_list in list(input_dict.values()):

        filtered_tokens_list = [word for word in token_list if word.lower() not in stop_words]
        concat_tokens = concat_tokens + filtered_tokens_list

    final_string = ' '.join(concat_tokens)
    return final_string, concat_tokens


# Make Wordcloud
def color_words(*args, **kwargs):
    return "rgb(0, 100, {})".format(random.randint(100, 255))

def generate_wordcloud(words_as_long_string):

    #warnings.filterwarnings("ignore")
    #wordcloud = WordCloud(background_color='black', width=800, height=400).generate(words_as_long_string)
    wordcloud = WordCloud(background_color='black').generate(words_as_long_string)

    img_io = io.BytesIO()
    wordcloud.to_image().save(img_io, format='PNG')
    img_io.seek(0)
    encoded_image = base64.b64encode(img_io.read()).decode("utf-8")

    #plt.imshow(wordcloud, interpolation='bilinear')
    #img =plt.imshow(wordcloud.recolor(color_func=color_words), interpolation="bilinear")
    #plt.axis("off")
    # Close the Matplotlib figure to prevent displaying it
    #plt.close()
    #plt.figure(figsize =(8,4))

    #encoded_image = base64.b64encode(img.tostring()).decode("utf-8")
    return encoded_image
    #return encoded_image

try:
    res = generate_wordcloud("I AM A BIG MAN")
    print("SUCCESS", type(res))
except:
    print("FAILURE")



# Define the Web App Layout
buttons_to_display_avgs = ['retweet_count', 'sentiment']
buttons_to_display_proportions = ["negative","neutral","positive"]
buttons_to_display_airlines = list(df_tokenize["airline"].unique())

dash_app.layout = dbc.Container([  html.Br(),
                                   dmc.Title('Proof of Concept Dashboard', color="blue", size="h1",  style={'text-align': 'center'}),
                                   html.Br(),

                                   html.Hr(),
                                   html.P("Analyze Exploratoire:", style={'text-align': 'center', 'font-size': '30px'}),

                                   html.P("Data table:", style={'font-size': '25px'}),
                                   dmc.Grid([
                                               dmc.Col( [ dash_table.DataTable( data=df.to_dict('records'), page_size=12, style_table={'overflowX': 'auto'} ) ],
                                                        span=6
                                                      )
                                            ]),  

                                   html.P("Average Values - Select a Variable to Analyze:", style={'font-size': '25px'}),
                                   dmc.Grid([
                                               dmc.RadioGroup( [dmc.Radio(i, value=i) for i in buttons_to_display_avgs],
                                                               id='radio-button-to-select-a-column',
                                                               value='sentiment', size="lg"
                                                             ),

                                               dmc.Col( [dcc.Graph(figure={}, id='graph-average-to-display')],
                                                        span=6
                                                       ),
                                            ]),

                                   html.P("Proportions - Select a Sentiment to Analyze:", style={'font-size': '25px'}),
                                   dmc.Grid([
                                       dmc.RadioGroup([dmc.Radio(i, value=i) for i in buttons_to_display_proportions],
                                                      id='radio-button-to-select-a-proportion',
                                                      value='negative', size="lg"
                                                      ),

                                       dmc.Col([dcc.Graph(figure={}, id='graph-proportions-to-display')],
                                               span=6
                                               ),
                                          ]),

                                   html.P("Words Frequency - Select an Airline:", style={'font-size': '25px'}),
                                   dmc.Grid([
                                       dmc.RadioGroup([dmc.Radio(i, value=i) for i in buttons_to_display_airlines],
                                                      id='radio-button-for-airlines',
                                                      value='Southwest', size="lg"
                                                      ),

                                       dmc.Col([dcc.Graph(figure={}, id='graph-words-frequency')],
                                               span=6
                                               ),
                                   ]),


                                   html.P("Wordcloud - Select an Airline:", style={'font-size': '25px'}),
                                   dmc.Grid([
                                       dmc.RadioGroup([dmc.Radio(i, value=i) for i in buttons_to_display_airlines],
                                                      id='radio-button-for-airlines2',
                                                      value='Southwest', size="lg"
                                                      ),

                                       dmc.Col([ dcc.Graph(figure={'data': [], 'layout': {}}, id='graph-wordcloud')],
                                               span=6
                                               ),
                                          ]),

                                   ], fluid=True

                            )



# SETUP CALLBACK: It Enables the use of control components for building the interaction
@callback(
    # By CONVENTION you MUST provide all Outputs first. "component_property" are the parameters whose values the interaction will change
    Output(component_id='graph-average-to-display', component_property='figure'),
    Input(component_id='radio-button-to-select-a-column', component_property='value'),

)
def update_avge_graph(col_chosen):

    fig = px.histogram(df, x='airline', y=col_chosen, histfunc='avg', title='Average Values by Airline')
    for category in df['airline'].unique():
        avg_value = df[df['airline'] == category][col_chosen].mean()
        fig.add_annotation(
            #text=f'Avg: {avg_value:.2f}',
            text=f'{avg_value:.2f}',
            x=category,
            y=avg_value,
            showarrow=False,
            font=dict(size=12),
            yshift=10
        )
    return fig

@callback(
    # By CONVENTION you MUST provide all Outputs first. "component_property" are the parameters whose values the interaction will change
    Output(component_id='graph-proportions-to-display', component_property='figure'),
    Input(component_id='radio-button-to-select-a-proportion', component_property='value'),

)
def update_poportion_graph(col_chosen):

    fig = px.bar(
        grouped[grouped["airline_sentiment"] == col_chosen],
        x="airline",
        y="proportion",
        text="proportion",
        title="Share of Selected Sentiment Out Of ALL Expressed for that Airline",
    )

    fig.update_traces(texttemplate="%{text:.2%}", textposition="inside")
    fig.update_layout(
        xaxis_title="Airline", yaxis_title="% Represented by Selected Sentiment"
    )
    return fig


@callback(
    # By CONVENTION you MUST provide all Outputs first. "component_property" are the parameters whose values the interaction will change
    Output(component_id='graph-words-frequency', component_property='figure'),
    Input(component_id='radio-button-for-airlines', component_property='value'),

)
def update_words_frequency_graph(chosen_airline):
    dict_of_users_tokens = {key: value for key, value in
                            zip(df_tokenize[df_tokenize["airline"] == chosen_airline]["tweet_id"],
                                df_tokenize[df_tokenize["airline"] == chosen_airline]["tokenized_tweet"]
                                )
                            }

    text, concatenated_tokens = make_string_for_wordcloud(dict_of_users_tokens)
    del text

    words_freq = pd.DataFrame(pd.Series(concatenated_tokens).value_counts()).reset_index()
    words_freq.columns = ["word", "count"]
    words_freq = words_freq.drop([0, 1])

    fig = px.bar(
        words_freq.head(30),
        x="word",
        y="count",
        text="count",
        title="Top Words frequency in Tweets for Airline",
    )

    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(
        xaxis_title="Words in Tweets", yaxis_title="Word Frequency"
    )
    return fig


@callback(
    # By CONVENTION you MUST provide all Outputs first. "component_property" are the parameters whose values the interaction will change
    Output(component_id='graph-wordcloud', component_property='figure'),
    Input(component_id='radio-button-for-airlines2', component_property='value'),

)
def update_wordcloud(chosen_airline):
    dict_of_users_tokens = {key: value for key, value in
                            zip(df_tokenize[df_tokenize["airline"] == chosen_airline]["tweet_id"],
                                df_tokenize[df_tokenize["airline"] == chosen_airline]["tokenized_tweet"]
                                )
                            }

    text, concatenated_tokens = make_string_for_wordcloud(dict_of_users_tokens)
    #del concatenated_tokens
    #print(text[0:50])

    encoded_image = generate_wordcloud(text)
    #print("worked", type(wordcloud_plot.to_dict()))df_tokenize
    print("worked")
    return {
        'data': [],
        'layout': {
            'images': [
                {
                    'source': 'data:image/png;base64,{}'.format(encoded_image),
                    'x': 0,
                    'y': 1,
                    'xref': 'paper',
                    'yref': 'paper',
                    'sizex': 1,
                    'sizey': 1,
                    'opacity': 1,
                    'xanchor': 'left',
                    'yanchor': 'top'
                }
            ],
            'xaxis': {
                'showgrid': False,
                'showticklabels': False,
                'zeroline': False,
            },
            'yaxis': {
                'showgrid': False,
                'showticklabels': False,
                'zeroline': False,
            },
        },
    }


if __name__ == '__main__':
    dash_app.run_server(debug=True)
#    app.run(debug=True)
#   app.run(host='0.0.0.0', port=5000, debug=True)
