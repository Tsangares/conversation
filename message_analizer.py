import pandas as pd
import math
import datetime
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
   

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

def plot_sentiment(conversation,names=['You','Me']):
    conversation = get_sentiment(conversation)
    #Splitting conversation
    you, me = split_conversation(conversation)
    
    #Averaging sentiment by week
    your_ave = you.groupby( pd.Grouper(key='time', freq='W-MON'))['sentiment'].median().reset_index().sort_values('time')
    my_ave = me.groupby( pd.Grouper(key='time', freq='W-MON'))['sentiment'].median().reset_index().sort_values('time')
    your_name = names[0]
    my_name = names[1]
    
    #Plotting
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(your_ave['time'], your_ave['sentiment'],label=your_name)
    ax.scatter(my_ave['time'], my_ave['sentiment'],label=my_name)
    ax.legend()
    ax.set_title("Sentiment over time")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax.set_xlabel("Time")
    ax.set_ylabel("Sentiment")
    plt.show()


if __name__ == "__main__":
    # Argparse input for conversation ID and filepath
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="The filepath to the signal messages")
    parser.add_argument("conversation_id", type=str, help="The conversation ID")
    args = parser.parse_args()
    CONVERSATION_ID = args.conversation_id
    
    df = get_signal_messages()
    df = get_conversation(df, CONVERSATION_ID)
    plot_sentiment(df)
