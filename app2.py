

import warnings
import os
import io
import random
import base64
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_bootstrap_components as dbc  # To define rows and columns on the page
import dash_mantine_components as dmc  # To define a grid on the page within which to insert dmc.Cols and define their width by assigning a number to the span property.
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import nltk
# Download "stop words" list!
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Download punctuation remover!
nltk.download("punkt")
import re
from wordcloud import WordCloud
import textblob
import snorkel
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model.label_model import LabelModel
import joblib

# New Method!!
from online_label_model import OnlineLabelModel
from sklearn.metrics import accuracy_score

SEED = 0



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
df["tweet_length"] = df["text"].apply(lambda x: len(x))
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
    

# Labeling Functions for prediction
model_sentiment140_to_load = os.path.join(
    os.getcwd(), "dash_trained_saved_models/pipe_sentiment140_tweets.pkl"
)
pipe_sentiment140_tweets = joblib.load(model_sentiment140_to_load)
@labeling_function()
def sklearn_nb_clf(text):
    # Decision of Classifier 1: SENTIMENT140
    return pipe_sentiment140_tweets.predict(text)[0]


model_imdb_to_load = os.path.join(
    os.getcwd(), "dash_trained_saved_models/pipe_imdb_reviews.pkl"
)
pipe_imdb_reviews = joblib.load(model_imdb_to_load)
@labeling_function()
def sklearn_nb_imdb_lf(text):
    # Decision of Classifier 2: IMDB REVIEWS
    return pipe_imdb_reviews.predict(text)[0]


textblob_pa_clf = textblob.Blobber(analyzer=textblob.sentiments.PatternAnalyzer())
@labeling_function()
def textblob_pa_lf(text):
    # Polarity of Classifier 3: RULES-BASED CLASSIFIER
    # polarity, subjectivity = textblob_pa_clf(text["text"]).sentiment
    polarity, subjectivity = textblob_pa_clf(str(text)).sentiment

    if polarity > 0.33:
        # if polarity > 0.5 and subjectivity > 0.5:
        return 2

    elif polarity > -0.33:
        # elif -0.5 < polarity < 0.5 and subjectivity > 0.5:
        return 1

    # when (subjectivity <= 0.5) OR when (subjectivity > 0.5 but polarity <= -0.5)
    return 0

lfs = [textblob_pa_lf, sklearn_nb_imdb_lf, sklearn_nb_clf]

# Function to Apply EXISTING Classifiers
def sequentially_apply_all_3_Classifiers(df_with_text, list_of_labeling_functions):

    applier = PandasLFApplier(list_of_labeling_functions)

    List_predicted_labels = applier.apply(df_with_text.to_frame(), progress_bar=False)
    return List_predicted_labels

#L_train = sequentially_apply_all_3_Classifiers(airline_text, lfs)
airline_text = df["text"]
lfs = [textblob_pa_lf, sklearn_nb_imdb_lf, sklearn_nb_clf]
L_train = sequentially_apply_all_3_Classifiers(airline_text, lfs)
y_true = df["sentiment"]


# Old technique: Label Model
label_model = LabelModel(cardinality=3)
label_model.fit(L_train, seed=SEED)


# Train Model for New Technique
def test_olm(k, alpha):
    olm = OnlineLabelModel(cardinality=3)

    batch_size = int(L_train.shape[0] / k)
    folds = {}
    y_fold = {}
    for i in range(k):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size
        folds[i] = L_train[start_index:end_index, :]
        y_fold[i] = y_true[start_index:end_index]

    # Counter
    fold_num = 0

    scores = {}
    lm_scores = {}

    for fold in folds.values():
        if fold_num == 0:
            olm.fit(fold, seed=SEED)
        else:
            olm.partial_fit(fold, alpha=alpha, update_tree=True, seed=SEED)

        # Get fold predictions
        preds = olm.predict(fold)
        lm_fold_preds = label_model.predict(fold)
        lm_fold_acc = accuracy_score(y_fold[fold_num], lm_fold_preds)
        lm_scores[fold_num] = lm_fold_acc

        acc = accuracy_score(y_fold[fold_num], preds)
        scores[fold_num] = acc
        fold_num += 1

    return lm_scores, scores, olm


# Results Graphs
optuna_number_of_batches = 916
optuna_alpha = 0.988411594

optuna_lm_scores_dict, optuna_scores_dict, olm_fitted = test_olm( k=optuna_number_of_batches, alpha=optuna_alpha)

def plot_data(scores, lm_scores):   
    lists = sorted(scores.items())
    lists_lm = sorted(lm_scores.items())
    x, y = zip(*lists)
    x_lm, y_lm = zip(*lists_lm)
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="red"),
            name="Accuracy of PER Batches Model",
        )
    )
    fig.add_traces(
        go.Scatter(
            x=x_lm,
            y=y_lm,
            mode="lines",
            line=dict(color="black"),
            name="Accuracy of Usual Snorkel Labeling Model",
        )
    )

    fig.update_layout(xaxis_title="Batch Number", yaxis_title="Accuracy")
    return fig


def accuracy_graph():

    alphas = [
            "Pattern Analyzer",
            "Transfer Learning from 'IMDB data'",
            "Combining the 3 Classifiers",
            "Transfer Learning from 'Sentiments140 data",
            "PER-BATCH Model",
        ]

    textblob_pa_lf_acc = 0.2901639344262295
    sklearn_nb_imdb_clf_acc = 0.5549863387978142
    lm_acc = 0.5638661202185792
    sklearn_nb_clf_acc = 0.6489754098360656
    olm_acc = 0.6827510917030567

    performance = [
            textblob_pa_lf_acc,
            sklearn_nb_imdb_clf_acc,
            lm_acc,
            sklearn_nb_clf_acc,
            olm_acc,
        ]

    fig = px.bar(
            x=performance,
            y=alphas,
            text=performance,
            title="Average Accuracy of Existing Methods Vs Per-Batch Method ",
        )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="inside")
    fig.update_layout(
            xaxis_title="Prediction Accuracy on Test Data", yaxis_title="Labeling Method"
        )

    return fig



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

                                   html.P("Tweets Length - Select an Airline:",  style={'font-size': '25px'}),
                                   dmc.Grid([
                                               dmc.RadioGroup([dmc.Radio(i, value=i) for i in buttons_to_display_airlines],
                                                              id='radio-button-for-airlines3',
                                                              value='Southwest', size="lg"
                                                              ),

                                               dmc.Col([dcc.Graph(figure={}, id='graph-tweet-length-to-display')],
                                                       span=6
                                                       ),
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

                                   html.Br(),
                                   html.Hr(),
                                   html.P("New Technique Results Comparisons:", style={'text-align': 'center', 'font-size': '30px'}),

                                   html.Hr(),
                                   html.P("Accuracy Graphs:", style={'font-size': '25px'}),
                                   dmc.Grid([
                                               dmc.Col([dcc.Graph(figure=accuracy_graph())],
                                                        span=6
                                                        ),

                                                dmc.Col([dcc.Graph(figure=plot_data(scores=optuna_scores_dict, lm_scores=optuna_lm_scores_dict))],
                                                         span=6
                                                        )

                                              ]),
                                   
                                   html.Hr(),
                                   html.P("Tweet Prediction:", style={'font-size': '25px'}),
                                   dmc.Grid([
                                                dmc.Col([dcc.Input(type='text', placeholder='Enter a Tweet here', size='lg',
                                                          value='', id='field-to-enter-tweet')],
                                                         span=4
                                                        ),

                                               # parameter "children" here is implicit
                                               dmc.Col([html.Div(id='prediction-of-tweet-sentiment', style={'font-family': 'Arial', 'font-size': '25px'})],
                                               span=6
                                                        ),
                                             ]),

                                   html.Br(),
                                   html.Br(),
                                   html.Br(),
                                   html.Br(),

                                   ], fluid=True

                            )



# SETUP CALLBACK: It Enables the use of control components for building the interaction
@callback(
    # By CONVENTION you MUST provide all Outputs first. "component_property" are the parameters whose values the interaction will change
    Output(component_id='graph-tweet-length-to-display', component_property='figure'),
    Input(component_id ='radio-button-for-airlines3', component_property='value'),
)
def update_tweet_length_graph(chosen_airline):

    if chosen_airline:

        fig = px.histogram(df[df["airline"] == chosen_airline], x="tweet_length",
                           text_auto='.0%', histnorm='probability',
                           title="Distribution of Tweets Length by Airline",
                           )
        fig.update_traces(textposition="inside")
        return fig

    else:
        return {'data': []}


@callback(
    # By CONVENTION you MUST provide all Outputs first. "component_property" are the parameters whose values the interaction will change
    Output(component_id='graph-average-to-display', component_property='figure'),
    Input(component_id='radio-button-to-select-a-column', component_property='value'),
)
def update_avge_graph(col_chosen):

    if col_chosen:

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

    else:
        return {'data': []}

@callback(
    # By CONVENTION you MUST provide all Outputs first. "component_property" are the parameters whose values the interaction will change
    Output(component_id='graph-proportions-to-display', component_property='figure'),
    Input(component_id='radio-button-to-select-a-proportion', component_property='value'),
)
def update_poportion_graph(col_chosen):

    if col_chosen:

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

    else:
        return {'data': []}


@callback(
    # By CONVENTION you MUST provide all Outputs first. "component_property" are the parameters whose values the interaction will change
    Output(component_id='graph-words-frequency', component_property='figure'),
    Input(component_id='radio-button-for-airlines', component_property='value'),
)
def update_words_frequency_graph(chosen_airline):

    if chosen_airline:

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

    else:
        return {'data': []}

@callback(
    # By CONVENTION you MUST provide all Outputs first. "component_property" are the parameters whose values the interaction will change
    Output(component_id='graph-wordcloud', component_property='figure'),
    Input(component_id='radio-button-for-airlines2', component_property='value'),
)
def update_wordcloud(chosen_airline):

    if chosen_airline:

        dict_of_users_tokens = {key: value for key, value in
                                zip(df_tokenize[df_tokenize["airline"] == chosen_airline]["tweet_id"],
                                    df_tokenize[df_tokenize["airline"] == chosen_airline]["tokenized_tweet"]
                                    )
                                }

        text, concatenated_tokens = make_string_for_wordcloud(dict_of_users_tokens)
        #del concatenated_tokens
        #print(text[0:50])

        try:
            encoded_image = generate_wordcloud(text)
            #print("Worked For WORDCLOUD!!!", type(encoded_image))
        except:
            print("Failed")

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

    else:
        return {'data': []}


@callback(
    # dash.dependencies.Output('text-output', 'children'),
    # [dash.dependencies.Input('text-input', 'value')]
    Output(component_id='prediction-of-tweet-sentiment', component_property='children'),
    Input(component_id='field-to-enter-tweet', component_property='value'),

)
def update_tweet_prediction(value):

    if value:
        tweet_to_predict = pd.Series(value)
        tweet_sources_preds = sequentially_apply_all_3_Classifiers(tweet_to_predict, lfs)
        new_technique_pred = olm_fitted.predict(tweet_sources_preds)[0]

        return f'The New Technique Sentiment Prediction of {value} is: {new_technique_pred} (0= Negatif, 1= Neutral, 2= Positive) '

    else:
        return f'No Tweet Entered!'






if __name__ == '__main__':
    dash_app.run_server(debug=True)
#   app.run(debug=True)
#   app.run(host='0.0.0.0', port=5000, debug=True)
