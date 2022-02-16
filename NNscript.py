# This script will contain the main pipeline regarding the training and test of three different neural network;
# respectively based on: scikit (sklearn.neural_network),tensorflow (keras) and neurolab.

# 1. Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import psutil
from sklearn.preprocessing import MinMaxScaler

# 2. Loading, pre-processing and data partition
ds = pd.read_csv('210526WeAdatasetRegression.csv')

ds.drop(ds.columns[[0, 1]], axis=1, inplace=True) # first two columns are subject identifier, i.e are not useful to predict asthma exacerbations
ds = pd.get_dummies(ds) # categorical variables are dummified

outcome = np.array(ds['ACTScore']) #variable to predict
print('outcome: ACTScore')
features = ds.drop('ACTScore', axis=1) #predictors of the model
print('features: ', list(features.columns))
features = np.array(features)

# training e test set partition
train_features, test_features, train_outcome, test_outcome = train_test_split(features, outcome,
                                                                              test_size=0.2,
                                                                              random_state=17)


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_outcome.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_outcome.shape)

# data scaling
scaler = MinMaxScaler()
scaler.fit(train_features)

train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)


