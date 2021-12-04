import pandas as pd
import os
import numpy as np
import time

# set rand seed
np.random.seed(1)

# load higgs boson dataset
print('loading dataset...')
higgs_dir = r'F:\Google Drive\umich\eecs545_machine_learning\final_project\HIGGS'
higgs_df = pd.read_csv(os.path.join(higgs_dir, 'HIGGS.csv'))

Y = higgs_df.iloc[:,0]
X = higgs_df.iloc[:,-7:]  # just use high level features
n = higgs_df.shape[0]
print('total num: {}'.format(n))
test_n = 500000
train_n = 1000

x_train_df = X.iloc[0:train_n,:]
y_train_df = Y.iloc[0:train_n]
print(x_train_df.shape)
print(y_train_df.shape)
x_test_df = X.iloc[-test_n:, :]
y_test_df = Y.iloc[-test_n:]

import xgboost

model_higgs = xgboost.XGBoostClassifier()
start = time.time()
model_higgs.fit(x_train_df, y_train_df, min_leaf=5, boosting_rounds=5, depth=5)
end = time.time()
print('original model time: {}'.format(end-start))

pred = model_higgs.predict(x_test_df)
acc = np.sum(pred == y_test_df.values)/len(pred)
print('accuracy = {}'.format(acc))

import xgboost2

model_higgs2 = xgboost2.XGBoostClassifier()
start = time.time()
model_higgs2.fit(x_train_df.values, y_train_df.values, min_num_leaf=5, boosting_rounds=5, max_depth=5)
end = time.time()
print('our model time: {}'.format(end-start))

pred = model_higgs2.predict(x_test_df.values)
acc = np.sum(pred == y_test_df.values)/len(pred)
print('accuracy = {}'.format(acc))