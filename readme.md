### To understand business context of the problem and solution, checkout the report "Pushpendra_Sharam_Capstone_Project_Report.pdf" 

### To see the sentiment analyzer app in action check out the video "Streamlit_App.mp4" 

__The requiremets.txt contains information about the virutal environment used in Python__ 

### The code folder contains all the relavant codes in notebooks and py files as described below:


   __Notebook 1: 1_Data_Cleaning_EDA__
    This notebook contains information about process of data collection for modelling. Different steps taken for data cleaning, and EDA are also described in this notebook. Following is the table of contents for this notebook.


   __Notebook 2: 2_Modelling__
    In the last notebook we carried out data cleaning and EDA to get some initial insights about our data. In this notebook we will use the cleaned data for modelling. Before the data can be fed to models, we have to transform text data to numeric data. Different steps taken to transform data are also described in this data. Finally, steps for training different models on the transformed data are described here.


   __Notebook 3: 3_Advanced_Modelling__
    In the last notebook we carried out data transformation to prepare data to be ready to fed into machine learning models. Different machine learning models were trained on the transformed data. The machine learning models took too much time for training on the the whole data set. Also some of the algorithms only accepts dense arrays for training. My laptop ran out of memory when I was converting the sparse matrix to dense arrays. Therefore, we will train some additional models only using a sample of data.
    In this notebook, we will take only 10 percent sample of the data and train some additional computationally intensive models on the sample of data.


   __Notebook 4: 4_Neural_Network_Modelling__
    In notebook 3, we tried some additional machine learning models on a sample of dataset.
    In this notebook, we will try some deep learning models on a sample of dataset. First a simple neural network was trained on the transformed data to predict sentiment of tweets. Second, word embeddings were used to train a neural network and logistic regression model.



  __Notebook 5: 5_Model_Testing__
    In the last notebook, we trained some deep learning models on a sample of dataset. This notebook describes process of testing a model using real world data.
    This process will be applied to all models, here I am showing the process for testing logistic regression model.
    In order to test the logistic regression model, tweets were scrapped from twitter and the model was used to predict the sentiment for the tweets. The tweets were scrapped for a specific brand to understand the public sentiment about that brand.
    Tweets were scrapped for last seven days and sentiment was predicted based on these tweets over last 7 days.


   __6_Streamlit_App.py__
    This files contains codes to deploy an app that scrapes data from twitter and predicts sentiment score using the developed machine learning models in the previous notebooks.

