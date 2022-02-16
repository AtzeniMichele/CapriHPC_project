"""
This script will contain the main pipeline regarding the training and test of two different neural network;
respectively based on: scikit (sklearn.neural_network),tensorflow (keras).
"""

# 1. Importing libraries
import signal
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import multiprocessing


# 2. functions for the NN parallel training using multiprocessing library
def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def train_model(NNlib):
    """
    This code is parallelized and runs on each process
    It trains a model with different layer sizes
    It saves the model and returns the score (R^2, MSE, MAE)
    """
    if NNlib == 'sklearn':
        from sklearn.neural_network import MLPRegressor
        mlp = MLPRegressor(hidden_layer_sizes=(50, 50), batch_size=10, max_iter=100)
        mlp.fit(train_features, train_outcome)

        predict_test = mlp.predict(test_features)
        RSS = ((test_outcome - predict_test) ** 2).sum()
        TSS = ((test_outcome - test_outcome.mean()) ** 2).sum()
        R_2 = 1 - (RSS / TSS)

        MAE = (abs(predict_test - test_outcome).sum()) / len(predict_test)
        MSE = (((predict_test - test_outcome) ** 2).sum()) / len(predict_test)

        return {'Library Ref': NNlib, 'test R^2': R_2, 'test MAE': MAE, 'test MSE': MSE}

    elif NNlib == 'tensorflow':
        import tensorflow as tf
        from tensorflow import keras

        learning_rate = 0.001
        epochs = 100
        batch_size = 10
        model = keras.Sequential([
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(1)
            ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mean_squared_error",
            metrics=tf.keras.metrics.MeanAbsoluteError()
        )

        model.fit(train_features, train_outcome, batch_size=batch_size, epochs=epochs)
        predict_test = model.predict(test_features)
        predict_test = np.array(predict_test)
        RSS = ((test_outcome - predict_test) ** 2).sum()
        TSS = ((test_outcome - test_outcome.mean()) ** 2).sum()
        R_2 = 1 - (RSS / TSS)

        MAE = (abs(predict_test - test_outcome).sum()) / len(predict_test)
        MSE = (((predict_test - test_outcome) ** 2).sum()) / len(predict_test)

        return {'Library Ref': NNlib, 'test R^2': R_2, 'test MAE': MAE, 'test MSE': MSE}


# 3. Loading, pre-processing and data partition
ds = pd.read_csv('210526WeAdatasetRegression.csv')

ds.drop(ds.columns[[0, 1]], axis=1,
        inplace=True)  # first two columns are subject identifier, i.e are not useful to predict asthma exacerbations
ds = pd.get_dummies(ds)  # categorical variables are dummified

outcome = np.array(ds['ACTScore'])  # variable to predict
print('outcome: ACTScore')
features = ds.drop('ACTScore', axis=1)  # predictors of the model
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

# 4. Run the model
start = timer()
num_workers = 3
NNlib = ['sklearn', 'tensorflow']
multiprocessing.set_start_method('fork', force=True)
pool = multiprocessing.Pool(num_workers, init_worker)
scores = pool.map(train_model, NNlib)
end = timer()

print(scores)
print('Elapsed time:', end - start, '[s]')
