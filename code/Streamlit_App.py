
from nbformat import write
import pandas as pd
import numpy as np
import streamlit as st

#set a title on the page
st.title('Uber pickup in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# create a text element and let the reader know the data is loading
data_load_state = st.text('Loading data...')

# Load 10,000 rows of data into the dataframe
data = load_data(10000)

# Notify the reader that the data was successfully loaded 
data_load_state.text('Done! (using st.cache)')

st.subheader('Raw data')
st.write(data)




