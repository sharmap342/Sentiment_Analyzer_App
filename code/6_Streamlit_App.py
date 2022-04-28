#import libraries 
import pandas as pd
import numpy as np
from datetime import datetime

#streamlit 
import streamlit as st
import joblib

#data scrapping from twitter
import tweepy as tw
import config #credentials for twitter API
 

#text processing 
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from  nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

#suppress warnings
import warnings
warnings.filterwarnings("ignore")

#Plotting
import matplotlib.pyplot as plt




#set a logo on the page
st.image('logo.png')

#Ask user to provide some text
text = st.text_input('Input text of your choice', value='SentiTweet App is Awesome!')

#tokenize text
snowball_stemmer = SnowballStemmer(language='english')
wordnet_stemmer = WordNetLemmatizer()
stop_words = stopwords.words('english')

tokenized_text = word_tokenize(text)
cleaned_text = [word for word in tokenized_text if word.casefold() not in stop_words]
cleaned_text = [snowball_stemmer.stem(word) for word in cleaned_text]


#transform the data 
bagofwords = joblib.load('count_vectorizer.pkl')
transformed_text = bagofwords.transform(cleaned_text)


#Ask user to provide a model 
option = st.selectbox('Select a model to predict sentiment', ('Logistic Regression', 
                        'Decision Tree', 'Random Forest'))


#Show the probability of being positive for the text
log_model = joblib.load('LogisticRegressoinModel.pkl')
prob = np.mean(log_model.predict_proba(transformed_text), axis=0)

if prob[1]>0.5:
    st.markdown("<h2 style='text-align: center; color: green;'>Message has Positive sentiment</h2>",
     unsafe_allow_html=True)
else:
    st.markdown("<h2 style='text-align: center; color: red;'>Message has Negative sentiment</h2>",
     unsafe_allow_html=True)



#Second part of the app where data is scrapped and plotted 
#plot a line chart from scrapped data if user based on user's experience 
if st.checkbox('Check sentiment for your brand on twitter', value=False):
    #scrape data from twitter based on user given date
    n_days = int(st.number_input('For how many days would you like to scrape tweets',
                                min_value=1, max_value=7, value=7))
    max_results = int(st.number_input('Number of tweets to be scraped for each day',
                                min_value=1, max_value=14, value=14))

    query = st.text_input('Select your brand', value='#Shopify')



    @st.cache(allow_output_mutation=True)
    def twitter_scrapper(n_days=7, query='#Shopify', max_results=14):
        """Scrapes data from twitter for given number of days
        Parameter
        ---------
        n_days: int, Number of days for which data to be scrapped
        query: str, tweets containing query will be scrapped
        max_results: int, Number of tweets to be scrapped for each day
        
        Returns
        -------
        A dataframe containing scrapped tweets"""

        #authenticate credentials 
        client = tw.Client(bearer_token=config.bearer_token)

        #create a list of timestamps for the last 7 days
        date_range = [datetime.now() - pd.Timedelta(days=x) for x in range(n_days)]
        date_range.append((datetime.now() - pd.Timedelta(days=n_days-1)).replace(hour=0, minute=0, second=0)) #starts the 1st day at 12 AM
        date_range.reverse() #sort in increasing order

        #get tweets for last n_days 
        tweets_dict = {}
        for i in range(len(date_range)-1):
            start_time = date_range[i]
            end_time = date_range[i+1]
            
            #get the tweets
            try:
                tweets = client.search_recent_tweets(
                                                    query=query, max_results=max_results, start_time=start_time,
                                                    end_time=end_time, tweet_fields=["created_at", "lang"])
                #store the list of english tweets 
                tweets_dict[end_time] = [tweet.text for tweet in tweets.data if tweet.lang == 'en']
            except:
                st.write(f'No matching tweets found for {end_time.date()}')

        #store tweets in a dataframe 
        tweets_df = pd.DataFrame.from_dict(tweets_dict, orient='index')
        return tweets_df


    tweets_df = twitter_scrapper(n_days, query, max_results)


    #Clean the scrapped data
    #First replace the None values with np.nan 
    tweets_df.replace([None], np.nan, inplace=True)

    @st.cache
    def text_clean(df):
        """Cleans user names and links from the text
        Parameters
        ----------
        df: Dataframe, dataframe that contains the columns to be cleaned
        Returns
        -------
        cleaned version of dataframe"""
        #define regular expression to find user names, links, and non alphanumeric characters from the text
        user_link = r"@\S+|https?:\S+|http?:\S+|[^A-Za-z0-9\s]+"
        
        for col in df:
            # remove user names, links and any other characters non alphanumeric characters 
            df[col] = df[col].str.replace(user_link, '')
            #remove white spaces from the text 
            df[col] = df[col].str.split().apply(lambda x: np.nan if x is np.nan else ' '.join(x)).str.strip() #gets rid of multiple whitespace
            df[col] = df[col].str.strip() #gets rid of spaces at the end 
            #replace empty strings in df with none
            df.replace(to_replace='', value=np.nan, inplace=True)

        return df

    tweets_clean = text_clean(tweets_df)

    #tokenize cleaned data 
    @st.cache
    def tokenize(df):
        """Tokenize data, removes stop words, and perform lemmitization
        Parameter
        ---------
        df: dataframe, Dataframe containing text to be tokenized
        Returns: 
        --------
        dataframe, dataframe with tokenized text"""
        for col in df:
            #tokenize words 
            df[col] = df[col].apply(lambda x: word_tokenize(x) if x is not np.nan else x)
            #remove stop words
            df[col] = df[col].apply(lambda x: [word for word in x if word.casefold() not in stop_words]
                                                if x is not np.nan else x)
            #perform stemming
            df[col] = df[col].apply(lambda x: [snowball_stemmer.stem(word) for word in x]
                                                if x is not np.nan else x)
        
        return df

    tweets_tokenized = tokenize(tweets_clean.copy())

    #tranform and predict probability for cleaned data 
    @st.cache
    def transform_proba(df, model=log_model):
        """transforms the data and predicts probability using a model
        Parameter
        ---------
        df: dataframe, contains tokenized text
        model: sklearn model, model for sentiment prediction
        Returns
        -------
        A dataframe containing probabilities over time"""
        temp_dict = {'pos_senti_proba': []}
        for index in df.index:
            #tranform data 
            test_transformed = bagofwords.transform(df.loc[index, :].dropna().apply(lambda x: ' '.join(x)))
            #predict average positive sentiment probability for each day
            temp_dict['pos_senti_proba'].append(np.mean(log_model.predict_proba(test_transformed), axis=0)[1])

        #store data in a dataframe and return it 
        return pd.DataFrame(temp_dict, index=df.index)

    result_df = transform_proba(tweets_tokenized)

    #Display probability chart with time 
    fig, ax = plt.subplots()
    ax.plot(result_df.index, result_df['pos_senti_proba'], '-ro')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sentiment score')
    ax.set_title('Sentiment score over time')
    fig.autofmt_xdate(rotation=45)
    st.pyplot(fig)







