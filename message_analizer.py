import pandas as pd
import math
import datetime
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from statsmodels.formula.api import ols
import statsmodels.api as sm


# To help find the conversation ID
def print_random_conversations():
    for msg in df[['body','conversationId']].sample(10):
        print(msg)

def get_signal_messages(filepath,months=12):
    df = pd.read_csv(filepath)
    df['time'] = df['received_at'].map(lambda x: datetime.datetime.fromtimestamp(x/1000.0))
    df['time'] = pd.to_datetime(pd.to_datetime(df['time']).dt.date)
    #Cut time
    df.dropna(subset=['body'],inplace=True)
    df = df[df['time'] > datetime.datetime.now() - datetime.timedelta(days=30*months)]
    print(df.columns)
    #df = df.get_sentiment(df)
    #df = df[['time','sentiment','positive','negative','neutral','body','type']]
    return df
    
def get_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()

    df['positive'] = df['body'].map(lambda x: analyzer.polarity_scores(x)['pos'])
    df['negative'] = df['body'].map(lambda x: analyzer.polarity_scores(x)['neg'])
    df['neutral'] = df['body'].map(lambda x: analyzer.polarity_scores(x)['neu'])
    df['sentiment'] = df['body'].map(lambda x: analyzer.polarity_scores(x)['compound'])
    return df
    
def get_conversation(df, _id):
    return df[df['conversationId'] == _id]
        
# Input Conversation Dataframe
# Return two dataframes, one for each person in the conversation
def split_conversation(convo):
    return convo[convo['type'] == "incoming"], convo[convo['type'] == "outgoing"]

def concatenate_day(df,time_column='time',concatenate_column='body'):
    return df.groupby('time')['body'].transform(lambda x: '; '.join(x).replace('\n','; ')).drop_duplicates()


def plot_sentiment_aggregate(conversation, names=['You','Me'], measure='mean', days='9D'):
    conversation = get_sentiment(conversation)
    conversation = conversation[conversation['sentiment'].notna()]
    #Splitting conversation
    you, me = split_conversation(conversation)
    you = you[you['sentiment'].notna()]
    me = me[me['sentiment'].notna()]

    #Averaging sentiment by day
    your_ave = you.groupby( pd.Grouper(key='time', freq=days))['sentiment'].mean().reset_index().sort_values('time')
    my_ave = me.groupby( pd.Grouper(key='time', freq=days))['sentiment'].mean().reset_index().sort_values('time')
    if measure=='var':
        your_ave = you.groupby( pd.Grouper(key='time', freq=days))['sentiment'].var().reset_index().sort_values('time')
        my_ave = me.groupby( pd.Grouper(key='time', freq=days))['sentiment'].var().reset_index().sort_values('time')
    if measure=='median':
        your_ave = your_ave[your_ave['sentiment'] > 0.01]
        my_ave = my_ave[my_ave['sentiment'] > 0.01]
        your_ave = you.groupby( pd.Grouper(key='time', freq=days))['sentiment'].median().reset_index().sort_values('time')
        my_ave = me.groupby( pd.Grouper(key='time', freq=days))['sentiment'].median().reset_index().sort_values('time')
    
    degree_fit = 2
    you_sentiment_fit = your_ave[your_ave['sentiment'].notna()]
    my_sentiment_fit = my_ave[your_ave['sentiment'].notna()]

    rlen=you_sentiment_fit['sentiment'].size
    fit_you = np.polyfit(range(rlen), you_sentiment_fit['sentiment'], deg=degree_fit)
    rlen=my_sentiment_fit['sentiment'].size
    fit_me = np.polyfit(range(rlen), my_sentiment_fit['sentiment'], deg=degree_fit)

    your_name = names[0]
    my_name = names[1]
    
    #Plotting
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(your_ave['time'], your_ave['sentiment'],label=your_name)
    ax.scatter(my_ave['time'], my_ave['sentiment'],label=my_name)
    ax.legend()
    ax.set_title("Sentiment over time")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    
    ax.plot(your_ave['time'][:rlen], np.polyval(fit_you, range(rlen)), color='blue')
    ax.plot(my_ave['time'][:rlen], np.polyval(fit_me, range(rlen)), color='orange')

    ax.set_xlabel("Time")
    ax.set_ylabel("Sentiment "+measure)
    plt.show()

    

def plot_sentiment_concatenated(conversation,names=['You','Me']):
    #Splitting conversation
    conversation = get_sentiment(conversation)
    you, me = split_conversation(conversation)
    
    #Getting names
    your_name = names[0]
    my_name = names[1]
    
    # print(you)
    #Plotting
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(you['time'], you['sentiment'],label=your_name)
    ax.scatter(me['time'], me['sentiment'],label=my_name)
    ax.legend()
    ax.set_title("Sentiment over time")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax.set_xlabel("Time")
    ax.set_ylabel("Sentiment")
    plt.show()

if __name__ == "__main__":
    # If there is an env then ignore the arguments
    import os
    if os.path.isfile(".env"):
        from dotenv import load_dotenv
        load_dotenv()
        CONVERSATION_ID = os.getenv("CONVERSATION_ID")
        FILEPATH = os.getenv("FILEPATH")
    else:
        # Argparse input for conversation ID and filepath
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("filepath", type=str, help="The filepath to the signal messages")
        parser.add_argument("conversation_id", type=str, help="The conversation ID")
        args = parser.parse_args()
        CONVERSATION_ID = args.conversation_id
        FILEPATH = args.filepath
    
    df = get_signal_messages(FILEPATH)
    df = get_conversation(df, CONVERSATION_ID)
    plot_sentiment_concatenated(df)
