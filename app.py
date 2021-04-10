"""
Created on Sat Apr 10 00:36:08 2021

@author: Victor Le
"""

# Import all the necessary packages for creating the input forms and the user endpoint
import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re

# Start my making the app and setting up the input form
app = FastAPI()
      
        
# Once we actually get the data (using the post method from the form) we need to do some preprocessing
data = pd.read_csv('./Sentiment.csv')
tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(data['text'].to_numpy())

# function for removing all the extra stuff from tweets 
def preprocess_data(text):
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text

# function for formatting the test sequence and preparing for input to the model
def my_pipeline(text):
    text_new = preprocess_data(text)
    X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
    X = pad_sequences(X, maxlen=28)
    return X

@app.get('/')
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}

# decorator 
@app.get('/predict', response_class = HTMLResponse)
def take_inp():
    # use HTML to generate a super simple form
    return '''
        <form method = "post">
        <input maxlength="28" name = "text" type = "text" value = "Input your sentence here!" />
        <input type="submit" />
        </form>'''
  
# put together the POST request 
@app.post('/predict')
def predict(text:str = Form(...)):
    clean_text = my_pipeline(text) #clean, and preprocess the text through pipeline
    loaded_model = tf.keras.models.load_model('sentiment.h5') #load the saved model 
    predictions = loaded_model.predict(clean_text) #predict the text
    sentiment = int(np.argmax(predictions)) #calculate the index of max sentiment
    probability = max(predictions.tolist()[0]) #calulate the probability
    
    if sentiment==0:
         t_sentiment = 'Negative' #set appropriate sentiment
    elif sentiment==1:
         t_sentiment = 'Neutral'
    elif sentiment==2:
         t_sentiment='Postive'
    return { #return the dictionary for endpoint
         "Your sentence": text,
         "Your predicted sentiment": t_sentiment,
         "Probability": probability
    }

