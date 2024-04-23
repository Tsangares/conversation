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
import seaborn as sns
from wordcloud import WordCloud


# To help find the conversation ID
def print_random_conversations():
    for msg in df[['body','conversationId']].sample(10):
        print(msg)

def get_signal_messages(filepath, id=None, months=12, time_setting='sent_at'):
    df = pd.read_csv(filepath)
    if type(id) == 'String':
        print('gotit')
        df = get_conversation(id)

    # Handle time
    df['time'] = df[time_setting].map(lambda x: datetime.datetime.fromtimestamp(x/1000.0))

    # Add some basic columns
    df['diffs']     = df['time'].sort_values().diff()
    df['weekday']   = df['time'].dt.day_name()
    df['hour']      = df['time'].dt.hour

    df['time'] = pd.to_datetime(pd.to_datetime(df['time']).dt.date)

    # df['time'] = df['sent_at'].map(lambda x: datetime.datetime.fromtimestamp(x/1000.0))
    
    # Cut time
    df.dropna(subset=['body'],inplace=True)
    df = df[df['time'] > datetime.datetime.now() - datetime.timedelta(days=30*months)]
    # print(df.columns)
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


def plot_sentiment_aggregate(conversation, names=['You','Me'], measure='mean', days='9D', your_name='You', my_name='Me'):
    conversation = get_sentiment(conversation)
    conversation = conversation[conversation['sentiment'].notna()]
    #Splitting conversation
    you, me = split_conversation(conversation)
    you = you[you['sentiment'].notna()]
    me = me[me['sentiment'].notna()]

    #Averaging sentiment by day
    your_ave = you.groupby( pd.Grouper(key='time', freq=days))['sentiment'].mean().reset_index().sort_values('time')
    my_ave = me.groupby( pd.Grouper(key='time', freq=days))['sentiment'].mean().reset_index().sort_values('time')
   
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
    
    your_ave['jtime']    = pd.DatetimeIndex(your_ave['time']).to_julian_date()
    my_ave['jtime']    = pd.DatetimeIndex(your_ave['time']).to_julian_date()

    ax = sns.regplot(x = "jtime", y = "sentiment", data = your_ave, label = your_name)
    ax = sns.regplot(x = "jtime", y = "sentiment", data = my_ave,   label = my_name)
    ax.set_ylabel('Senteiment')
    ax.set_xlabel('Date')

    xtick_val = 4
    if len(your_ave['jtime']) > 10:
        xtick_val = len(your_ave['jtime'])//3

    ax.set_xticks(your_ave['jtime'][::xtick_val])
    plt.legend()
    _ = ax.set_xticklabels(your_ave['time'].dt.date[::xtick_val], rotation=45, ha='right')
    ax.set_title('Sentiment over time')
    plt.show()

def plot_sentiment_time_of_day(conversation, names=['You','Me'], measure='mean', days='9D', your_name='You', my_name='Me'):
    conversation = get_sentiment(conversation)
    conversation = conversation[conversation['sentiment'].notna()]
    #Splitting conversation
    you, me = split_conversation(conversation)
    you = you[you['sentiment'].notna() & you['sentiment'] != 0]
    me = me[me['sentiment'].notna() & me['sentiment'] != 0]
    you_hour = you.groupby('hour')['sentiment'].median()
    me_hour  = me.groupby('hour')['sentiment'].median()
    new_df      = pd.DataFrame(index=you_hour.index, data={'you': you_hour.values, 'me': me_hour.values})
    new_df.plot.bar(title='Sentiment by time of day', ylabel='Sentiment score', xlabel='Hour in day')
    
    plt.legend()
    plt.show()

def plot_time_between_messages(df):
    df_you, df_me = split_conversation(df)

    def make_diff_df(in_df):
        df_diff             = in_df.groupby(pd.Grouper(key='time', freq='35D'))['diffs'].median().reset_index().sort_values('diffs')
        df_diff['diffs']    = df_diff['diffs'].astype('timedelta64[s]')  / np.timedelta64(1, 's')
        df_diff             = df_diff[df_diff['diffs'].notna()]
        df_diff['diffs']    = df_diff['diffs'].astype('str').astype('double').astype(int)
        df_diff             = df_diff[['diffs','time']].dropna()
        df_diff['jtime']    = pd.DatetimeIndex(df_diff['time']).to_julian_date()
        return df_diff
    you_diff_df = make_diff_df(df_you)
    me_diff_df  = make_diff_df(df_me)

    ax = sns.regplot(x = "jtime", y = "diffs", data = you_diff_df, label='You')
    ax = sns.regplot(x = "jtime", y = "diffs", data = me_diff_df, label='Me')
    plt.legend()
    ax.set_xticks(you_diff_df['jtime'])
    _ = ax.set_xticklabels(you_diff_df['time'].dt.date, rotation=45, ha='right')
    ax.set_title('Time between texts')
    plt.show()

def wordclouds(conversation):

    stop = stopwords.words('english')

    df_you, df_me = split_conversation(conversation)

    words_me  = df_me['body'].str.split(expand=True).stack()
    words_you = df_you['body'].str.split(expand=True).stack()

    words_me = words_me.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop) and 'http' not in word]))
    words_you = words_you.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop) and 'http' not in word]))

    wordcloud2 = WordCloud(collocations = False, background_color = 'white').generate(' '.join(list(words_you)))
    # Generate plot
    plt.imshow(wordcloud2)
    plt.axis("off")
    plt.title('Your word cloud')
    plt.show()
    wordcloud2 = WordCloud(collocations = False, background_color = 'white').generate(' '.join(list(words_me)))
    # Generate plot
    plt.imshow(wordcloud2)
    plt.axis("off")
    plt.title('My word cloud')
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
