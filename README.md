# SentimentML

# Project Goals 
This project was focused not so much on training and fine-tuning a model, but rather packaging it for deployment for end users. The dataset being used for this project is the *First GOP Debate Twitter Sentiment* data set from [Kaggle](https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment). The goal of the deep learning model is to label a tweet as either having a *positive*, *neutral*, or *negative* sentiment. Most of the model training is performed in [DeployML.ipynb](https://github.com/victle/SentimentML/blob/main/DeployML.ipynb), and will be summarized below alongside the extra steps needed to package and deploy the model. 

# Loading and pre-processing Data 
## Packages 
In Python 3.9.2, some packages that we'll be using are:
* Model Training
  * Tensorflow (version 2.0.0) 
    * NOTE: For deployment in Heroku, tensorflow-cpu should be installed to reduce the size of the application
  * Pandas
  * Numpy (version 1.19.2)
  * sklearn
  * re
  * h5py
* Deployment
  * fastapi
  * uvicorn
  * python-multipart
  * pydantic

## Data Processing
Explore [DeployML.ipynb](https://github.com/victle/SentimentML/blob/main/DeployML.ipynb) to get an in-depth look at how the tweets are cleaned up, tokenized, and transformed in a equal-sized sequences, as well as see the architecture and training of a basic model.

**Because the focus of the project was deployment, model has not been fine-tuned. Increased performance is planned but TBD.**

# Visualizations 
## Tweets being cleaned up
![image](https://user-images.githubusercontent.com/26015263/114282124-0e991600-9a10-11eb-9444-8c395b378845.png) ![image](https://user-images.githubusercontent.com/26015263/114282129-148ef700-9a10-11eb-918e-5a5df45677aa.png)

## Distribution of the sentiments
![image](https://user-images.githubusercontent.com/26015263/114281845-b9103980-9a0e-11eb-9c38-80f29578781a.png)

## Model performance (will be improved in future)
![image](https://user-images.githubusercontent.com/26015263/114282143-2ec8d500-9a10-11eb-94b6-fcf54b3c43d0.png)

# Deployment 
Look in [app.py](https://github.com/victle/SentimentML/blob/main/app.py). Using `FastAPI()`, a simple HTML form can be created, such that users can input and submit a phrase. The model will then spit out a prediction as well as the prediction probability (essentially, confidence in the estimate). The app can then be hosted locally using a uvicorn server, with `uvicorn app:app --reload`. Finally, Heroku can be setup to connect with a Github repository in order to deploy an app online. 

[You can access my super simple version of it here!](https://sentimentml.herokuapp.com/predict) Performance is not great at the moment, until I can go back and improve model performance. 

## A simple demo
![image](https://user-images.githubusercontent.com/26015263/114282384-a3504380-9a11-11eb-956a-845dfb9cff56.png)
![image](https://user-images.githubusercontent.com/26015263/114282389-ae0ad880-9a11-11eb-98a0-dd51e14b430c.png)
