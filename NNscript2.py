# 1. Importing libraries
import os
import signal
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import multiprocessing
import os


# 2. functions for the NN parallel training using multiprocessing library
def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def train_model(seed):
    import tensorflow as tf
    from tensorflow import keras
    from sklearn import metrics

    # training e test set partitions
    train_features, test_features, train_outcome, test_outcome = train_test_split(features, outcome,
                                                                                  test_size=0.2,
                                                                                  random_state=seed)

    # data scaling
    scaler = StandardScaler()
    scaler.fit(train_features)

    target_scaler = StandardScaler()
    target_scaler.fit(train_outcome)

    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    train_outcome = target_scaler.transform(train_outcome)
    test_outcome = target_scaler.transform(test_outcome)

    learning_rate = 0.001
    epochs = 1000
    batch_size = 10
    model = keras.Sequential([
        keras.layers.Dense(units=50, input_dim=18, activation='relu'),
        keras.layers.Dense(units=50, activation='relu'),
        keras.layers.Dense(1, kernel_initializer='normal')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error',
        metrics=tf.keras.metrics.MeanAbsoluteError()
    )

    model.fit(train_features, train_outcome, batch_size=batch_size, epochs=epochs)

    predict_test = model.predict(test_features)
    # a rough method to extract prediction values ( list of lists) in order to compute linear mathematical operations
    # ans = np.empty(len(predict_test), dtype=float)
    # for i in range(len(predict_test)):
    #     ind = predict_test[i]
    #     for j in ind:
    #         ans[i] = j
    # predict_test = ans

    # RSS = ((test_outcome - predict_test) ** 2).sum()
    # TSS = ((test_outcome - test_outcome.mean()) ** 2).sum()
    # R_2 = 1 - (RSS / TSS)
    #
    # MAE = (abs(predict_test - test_outcome).sum()) / len(predict_test)
    # MSE = (((predict_test - test_outcome) ** 2).sum()) / len(predict_test)

    MAE = metrics.mean_absolute_error(test_outcome, predict_test)
    MSE = metrics.mean_squared_error(test_outcome, predict_test)
    R_2 = metrics.r2_score(test_outcome, predict_test)

    return {'seed': seed, 'test R^2': R_2, 'test MAE': MAE, 'test MSE': MSE}


# 3. Loading, pre-processing and data partition
#ds = pd.read_csv('/capri_project/210526WeAdatasetRegression.csv')
ds = pd.read_csv('210526WeAdatasetRegression.csv')

ds.drop(['UserNo.', 'UserID', 'SmokingHabit'], axis=1,
        inplace=True)  # first two columns are subject identifier, i.e are not useful to predict asthma exacerbations
ds = pd.get_dummies(ds, drop_first=True)  # categorical variables are dummified

# filtered_columns = ds.dtypes[ds.dtypes == np.object]
# list_columns = list(filtered_columns.index)
# for column in list_columns:
#     ds[column] = LabelEncoder().fit_transform(ds[column])

outcome = np.array(ds['ACTScore'])
outcome = outcome.reshape(-1, 1)

print('outcome: ACTScore')
features = ds.drop('ACTScore', axis=1)  # predictors of the model
print('features: ', list(features.columns))
features = np.array(features)

# 4. Run the model
start = timer()
num_workers = os.cpu_count()
seeds = range(10)
multiprocessing.set_start_method('fork', force=True)
pool = multiprocessing.Pool(num_workers, init_worker)
scores = pool.map(train_model, seeds)
end = timer()

print(scores)

cv_r2 = [list(scores[i].values())[1] for i in range(len(scores))]
print('R^2 mean: {}  and standard deviation: {}'.format(np.mean(cv_r2), np.std(cv_r2)))

cv_mae = [list(scores[i].values())[2] for i in range(len(scores))]
print('MAE mean: {}  and standard deviation: {}'.format(np.mean(cv_mae), np.std(cv_mae)))

cv_mse = [list(scores[i].values())[3] for i in range(len(scores))]
print('MSE mean: {}  and standard deviation: {}'.format(np.mean(cv_mse), np.std(cv_mse)))

print('Elapsed time:', end - start, '[s]')
print('Number of workers (e.g. CPUs):', num_workers)
